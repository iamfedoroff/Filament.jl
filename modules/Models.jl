module Models

import CuArrays
import CUDAnative
import CUDAdrv
using PyCall
@pyimport scipy.constants as sc
@pyimport matplotlib.pyplot as plt

import Units
import Grids
import Fields
import Media
import Plasmas
import HankelGPU
import Fourier
import FourierGPU
import RungeKuttasGPU
import Guards

const C0 = sc.c   # speed of light in vacuum
const EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = sc.mu_0   # the magnetic constant [N/A^2]
const QE = sc.e   # elementary charge [C]
const ME = sc.m_e   # electron mass [kg]
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32
const ComplexGPU = Complex64


struct Model
    KZ_gpu :: CuArrays.CuArray{ComplexGPU, 2}
    QZ_gpu :: CuArrays.CuArray{ComplexGPU, 2}
    Rk_gpu :: FloatGPU
    Rr_gpu :: FloatGPU
    Hramanw_gpu :: CuArrays.CuArray{ComplexGPU, 1}
    Rp_gpu :: CuArrays.CuArray{ComplexGPU, 1}
    Ra_gpu :: CuArrays.CuArray{ComplexGPU, 1}
    phi_kerr :: Float64
    phi_plasma :: Float64
    guard :: Guards.GuardFilter
    RKGPU :: Union{RungeKuttasGPU.RungeKutta2, RungeKuttasGPU.RungeKutta3,
                   RungeKuttasGPU.RungeKutta4}
    keys :: Dict{String, Any}
end


function Model(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
               medium::Media.Medium, plasma::Plasmas.Plasma,
               keys::Dict{String, Any})
    # Guards -------------------------------------------------------------------
    rguard_width = keys["rguard_width"]
    tguard_width = keys["tguard_width"]
    kguard = keys["kguard"]
    wguard = keys["wguard"]
    guard = Guards.GuardFilter(unit, grid, medium,
                               rguard_width, tguard_width, kguard, wguard)

    # Runge-Kutta --------------------------------------------------------------
    RKORDER = keys["RKORDER"]
    RKGPU = RungeKuttasGPU.RungeKutta(RKORDER, grid.Nr, grid.Nw)

    # Linear propagator --------------------------------------------------------
    KPARAXIAL = keys["KPARAXIAL"]

    beta = Media.beta_func.(medium, grid.w * unit.w)
    KZ = zeros(Complex128, (grid.Nr, grid.Nw))
    if KPARAXIAL != 0
        for j=1:grid.Nw
            if beta[j] != 0.
                for i=1:grid.Nr
                    KZ[i, j] = beta[j] - (grid.k[i] * unit.k)^2 / (2. * beta[j])
                end
            end
        end
    else
        for j=1:grid.Nw
            for i=1:grid.Nr
                KZ[i, j] = sqrt(beta[j]^2 - (grid.k[i] * unit.k)^2 + 0im)
            end
        end
    end

    vf = Media.group_velocity(medium, field.w0)   # frame velocity
    for j=1:grid.Nw
        for i=1:grid.Nr
            KZ[i, j] = (KZ[i, j] - grid.w[j] * unit.w / vf) * unit.z
        end
    end

    @. KZ = conj(KZ)

    # Nonlinear propagator -----------------------------------------------------
    QPARAXIAL = keys["QPARAXIAL"]

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. MU0 * mu * (grid.w * unit.w)^2 / 2. * unit.z / Eu

    QZ = zeros(Complex128, (grid.Nr, grid.Nw))
    if QPARAXIAL != 0
        for j=1:grid.Nw
            if beta[j] != 0.
                for i=1:grid.Nr
                    QZ[i, j] = Qfactor[j] / beta[j]
                end
            end
        end
    else
        for j=1:grid.Nw
            for i=1:grid.Nr
                kzij = sqrt(beta[j]^2 - (grid.k[i] * unit.k)^2 + 0im)
                if kzij != 0.
                    QZ[i, j] = Qfactor[j] / kzij
                end
            end
        end
    end

    @. QZ = conj(QZ)

    # Kerr nonlinearity --------------------------------------------------------
    Rk = Rk_func(unit, field, medium)
    phi_kerr = phi_kerr_func(unit, field, medium)

    # Stimulated Raman nonlinearity --------------------------------------------
    RAMAN = keys["RAMAN"]

    graman = medium.graman

    if RAMAN != 0
        Rk = (1. - graman) * Rk
        Rr = graman * Rk
    else
        Rr = 0.
    end

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put H(t)
    # into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. medium.raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * grid.dt * unit.t

    if abs(1. - sum(Hraman)) > 1e-6
        print("WARNING: The integral of Raman response function should be" *
              " normalized to 1.\n")
    end

    @. Hraman = Hraman * guard.T   # temporal filter
    Hraman = ifftshift(Hraman)
    Hramanw = Fourier.rfft1d(grid.FT, Hraman)   # time -> frequency

    # Plasma nonlinearity ------------------------------------------------------
    Rp = Rp_func(unit, grid, field, medium, plasma)
    @. Rp = conj(Rp)
    phi_plasma = phi_kerr_func(unit, field, medium)

    # Losses due to multiphoton ionization -------------------------------------
    Ra = Ra_func(unit, grid, field, medium)
    @. Ra = conj(Ra)

    # GPU:
    KZ_gpu = CuArrays.cu(convert(Array{ComplexGPU, 2}, KZ))
    QZ_gpu = CuArrays.cu(convert(Array{ComplexGPU, 2}, QZ))
    Rk_gpu = FloatGPU(Rk)
    Rr_gpu = FloatGPU(Rr)
    Hramanw_gpu = CuArrays.cu(convert(Array{ComplexGPU, 1}, Hramanw))
    Rp_gpu = CuArrays.cu(convert(Array{ComplexGPU, 1}, Rp))
    Ra_gpu = CuArrays.cu(convert(Array{ComplexGPU, 1}, Ra))

    return Model(KZ_gpu, QZ_gpu, Rk_gpu, Rr_gpu, Hramanw_gpu, Rp_gpu, Ra_gpu,
                 phi_kerr, phi_plasma, guard, RKGPU, keys)
end


function adaptive_dz(model::Model, AdaptLevel::Float64, I::Float64,
                     rho::Float64)
    if model.keys["KERR"] != 0
        dz_kerr = model.phi_kerr / I * AdaptLevel
    else
        dz_kerr = Inf
    end

    if (model.keys["PLASMA"] != 0) & (rho != 0.)
        dz_plasma = model.phi_plasma / rho * AdaptLevel
    else
        dz_plasma = Inf
    end

    dz = min(dz_kerr, dz_plasma)
    return dz
end


function zstep(dz::Float64, grid::Grids.Grid, field::Fields.Field,
               plasma::Plasmas.Plasma, model::Models.Model)


    function func_gpu!(S_gpu::CuArrays.CuArray{ComplexGPU, 2},
                       res_gpu::CuArrays.CuArray{ComplexGPU, 2})
        Ec_gpu = CuArrays.cuzeros(ComplexGPU, grid.Nt)
        Er_gpu = CuArrays.cuzeros(FloatGPU, grid.Nt)
        Sr_gpu = CuArrays.cuzeros(ComplexGPU, grid.Nw)
        Ftmp_gpu = CuArrays.cuzeros(FloatGPU, grid.Nt)
        Stmp_gpu = CuArrays.cuzeros(ComplexGPU, grid.Nw)
        resi_gpu = CuArrays.cuzeros(ComplexGPU, grid.Nw)
        rhoi_gpu = CuArrays.cuzeros(FloatGPU, grid.Nt)
        Kdrhoi_gpu = CuArrays.cuzeros(FloatGPU, grid.Nt)
        zeros_gpu = CuArrays.cuzeros(ComplexGPU, grid.Nw)

        for i=1:grid.Nr
            equal1!(Sr_gpu, i, S_gpu)   # Sr_gpu = S_gpu[i, :]
            FourierGPU.spectrum_real_to_signal_analytic!(grid.FTGPU, Sr_gpu, Ec_gpu)
            @inbounds @. Er_gpu = real(Ec_gpu)

            @inbounds @. resi_gpu = zeros_gpu

            # Kerr nonlinearity:
            if model.keys["KERR"] != 0
                if model.keys["THG"] != 0
                    @inbounds @. Ftmp_gpu = Er_gpu^3
                else
                    @inbounds @. Ftmp_gpu = 3. / 4. * abs2(Ec_gpu) * Er_gpu
                end
                Guards.apply_temporal_filter!(model.guard, Ftmp_gpu)
                FourierGPU.rfft1d!(grid.FTGPU, Ftmp_gpu, Stmp_gpu)   # time -> frequency

                @inbounds @. resi_gpu = resi_gpu + model.Rk_gpu * Stmp_gpu
            end

            # Stimulated Raman nonlinearity:
            if model.keys["RAMAN"] != 0
                if model.keys["RTHG"] != 0
                    @inbounds @. Ftmp_gpu = Er_gpu^2
                else
                    @inbounds @. Ftmp_gpu = 3. / 4. * abs2(Ec_gpu)
                end
                Guards.apply_temporal_filter!(model.guard, Ftmp_gpu)
                FourierGPU.convolution!(grid.FTGPU, model.Hramanw_gpu, Ftmp_gpu)
                @inbounds @. Ftmp_gpu = Ftmp_gpu * Er_gpu
                Guards.apply_temporal_filter!(model.guard, Ftmp_gpu)
                FourierGPU.rfft1d!(grid.FTGPU, Ftmp_gpu, Stmp_gpu)   # time -> frequency

                @inbounds @. resi_gpu = resi_gpu + model.Rr_gpu * Stmp_gpu
            end

            # Plasma nonlinearity:
            if model.keys["PLASMA"] != 0
                equal1!(rhoi_gpu, i, rho_gpu)   # rhoi_gpu = rho_gpu[i, :]
                @inbounds @. Ftmp_gpu = rhoi_gpu * Er_gpu
                Guards.apply_temporal_filter!(model.guard, Ftmp_gpu)
                FourierGPU.rfft1d!(grid.FTGPU, Ftmp_gpu, Stmp_gpu)   # time -> frequency

                @inbounds @. resi_gpu = resi_gpu + model.Rp_gpu * Stmp_gpu
            end

            # Losses due to multiphoton ionization:
            if model.keys["ILOSSES"] != 0
                equal1!(Kdrhoi_gpu, i, Kdrho_gpu)   # Kdrhoi_gpu = Kdrho_gpu[i, :]

                if model.keys["IONARG"] != 0
                    @inbounds @. Ftmp_gpu = abs2(Ec_gpu)
                else
                    @inbounds @. Ftmp_gpu = Er_gpu^2
                end

                safe_inverse!(Ftmp_gpu)
                @inbounds @. Ftmp_gpu = Kdrhoi_gpu * Ftmp_gpu * Er_gpu
                Guards.apply_temporal_filter!(model.guard, Ftmp_gpu)
                FourierGPU.rfft1d!(grid.FTGPU, Ftmp_gpu, Stmp_gpu)   # time -> frequency

                @inbounds @. resi_gpu = resi_gpu + model.Ra_gpu * Stmp_gpu
            end

            equal2!(res_gpu, i, resi_gpu)   # res_gpu[i, :] = resi_gpu
        end

        # Nonparaxiality:
        if model.keys["QPARAXIAL"] != 0
            @inbounds @. res_gpu = -1im * model.QZ_gpu * res_gpu
        else
            print("STOP!\n")
            quit()
            for j=1:grid.Nw
                res[:, j] = Hankel.dht(grid.HT, res[:, j])
            end
            @inbounds @. res = -1im * model.QZ * res
            @inbounds @. res = res * model.guard.K   # angular filter
            for j=1:grid.Nw
                res[:, j] = Hankel.idht(grid.HT, res[:, j])
            end
        end

        return nothing
    end


    # Calculate plasma density -------------------------------------------------
    if (model.keys["PLASMA"] != 0) | (model.keys["ILOSSES"] != 0)
        Plasmas.free_charge(plasma, grid, field)
    end

    dz_gpu = FloatGPU(dz)

    # Field -> temporal spectrum -----------------------------------------------
    FourierGPU.rfft2d!(grid.FTGPU, field.E_gpu, field.S_gpu)

    # Nonlinear propagator -----------------------------------------------------
    if (model.keys["KERR"] != 0) | (model.keys["PLASMA"] != 0) |
       (model.keys["ILOSSES"] != 0)
        rho_gpu = CuArrays.cu(convert(Array{FloatGPU, 2}, plasma.rho))
        Kdrho_gpu = CuArrays.cu(convert(Array{FloatGPU, 2}, plasma.Kdrho))

        RungeKuttasGPU.RungeKutta_calc!(model.RKGPU, field.S_gpu, dz_gpu, func_gpu!)
    end

    # Linear propagator --------------------------------------------------------
    HankelGPU.dht!(grid.HTGPU, field.S_gpu)
    @inbounds @. field.S_gpu = field.S_gpu * exp_cuda(model.KZ_gpu * dz_gpu)
    Guards.apply_angular_filter!(model.guard, field.S_gpu)
    HankelGPU.idht!(grid.HTGPU, field.S_gpu)

    # Temporal spectrum -> field -----------------------------------------------
    Guards.apply_spectral_filter!(model.guard, field.S_gpu)
    FourierGPU.spectrum_real_to_signal_analytic_2d!(grid.FTGPU, field.S_gpu, field.E_gpu)
    Guards.apply_spatial_filter!(model.guard, field.E_gpu)
    Guards.apply_temporal_filter!(model.guard, field.E_gpu)

    # Collect field from GPU:
    field.E = convert(Array{Complex128, 2}, CuArrays.collect(field.E_gpu))

    return nothing
end


function Rk_func(unit::Units.Unit, field::Fields.Field, medium::Media.Medium)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    chi3 = Media.chi3_func(medium, field.w0)
    R = EPS0 * chi3 * Eu^3
    return R
end


function Rp_func(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                 medium::Media.Medium, plasma::Plasmas.Plasma)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    nuc = plasma.nuc
    MR = plasma.mr * ME   # reduced mass of electron and hole (effective mass)
    R = zeros(Complex128, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            R[i] = 1im / (grid.w[i] * unit.w) *
                   QE^2 / MR / (nuc - 1im * (grid.w[i] * unit.w)) *
                   unit.rho * Eu
        end
    end
    return R
end


function Ra_func(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                 medium::Media.Medium)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)

    R = zeros(Complex128, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            R[i] = 1im / (grid.w[i] * unit.w) *
                   HBAR * field.w0 * unit.rho / (unit.t * Eu)
        end
    end
    return R
end


"""Kerr phase factor for adaptive z step."""
function phi_kerr_func(unit::Units.Unit, field::Fields.Field,
                       medium::Media.Medium)
    w0 = field.w0
    n0 = real(Media.refractive_index(medium, field.w0))
    k0 = Media.k_func(medium, w0)
    Eu = Units.E(unit, n0)
    mu = medium.permeability(w0)
    chi3 = Media.chi3_func(medium, field.w0)
    Rk0 = mu * w0^2 / (2. * C0^2) * chi3 * Eu^2 * unit.z

    if real(Rk0) != 0.
        phi_real = k0 / (3. / 4. * abs(real(Rk0)))
    else
        phi_real = Inf
    end

    if imag(Rk0) != 0.
        phi_imag = k0 / (3. / 4. * abs(imag(Rk0)))
    else
        phi_imag = Inf
    end

    phi = min(phi_real, phi_imag)
    return phi
end


"""Plasma phase factor for adaptive z step."""
function phi_plasma(unit::Units.Unit, field::Fields.Field, medium::Media.Medium)
    w0 = field.w0
    k0 = Media.k_func(medium, w0)
    nuc = medium.nuc
    mu = medium.permeability(w0)
    MR = medium.mr * ME   # reduced mass of electron and hole (effective mass)
    Rp0 = 0.5 * MU0 * mu * w0 / (nuc - 1im * w0) * QE^2 / MR * unit.rho * unit.z

    if real(Rp0) != 0.
        phi_real = k0 / abs(real(Rp0))
    else
        phi_real = Inf
    end

    if imag(Rp0) != 0.
        phi_imag = k0 / abs(imag(Rp0))
    else
        phi_imag = Inf
    end

    phi = min(phi_real, phi_imag)
    return phi
end


"""
Calculates "exp(-1im * x)" on GPU.

Unfortunately, "CUDAnative.exp.(x)" function does not work with complex
arguments. To solve the issue, I use Euler's formula:
    exp(-1im * x) = (cos(xr) - 1im * sin(xr)) * exp(xi),
where xr = real(x) and xi = imag(x).
"""
function exp_cuda(x::ComplexGPU)
    xr = real(x)
    xi = imag(x)
    return (CUDAnative.cos(xr) - 1im * CUDAnative.sin(xr)) * CUDAnative.exp(xi)
end


function safe_inverse!(a_gpu::CuArrays.CuArray{FloatGPU, 1})

    function kernel(a::CUDAnative.CuDeviceArray{Float32,1})
        i = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
        if i <= length(a)
            if a[i] >= FloatGPU(1e-30)
                a[i] = FloatGPU(1.) / a[i]
            else
                a[i] = FloatGPU(0.)
            end
        end
        return nothing
    end

    dev = CUDAnative.CuDevice(0)
    N = length(a_gpu)
    MAX_THREADS = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    threads = min(N, MAX_THREADS)
    blocks = Int(ceil(N / threads))

    @CUDAnative.cuda (blocks, threads) kernel(a_gpu)
    return nothing
end


# function equal1!(b_gpu::CuArrays.CuArray{ComplexGPU, 1},
#                  i::Int64,
#                  a_gpu::CuArrays.CuArray{ComplexGPU, 2})
function equal1!(b_gpu, i::Int64, a_gpu)

    function kernel(b, i, a)
        j = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
        if j <= length(b)
            b[j] = a[i, j]
        end
        return nothing
    end

    dev = CUDAnative.CuDevice(0)
    N = length(b_gpu)
    MAX_THREADS = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    threads = min(N, MAX_THREADS)
    blocks = Int(ceil(N / threads))

    @CUDAnative.cuda (blocks, threads) kernel(b_gpu, i, a_gpu)
    return nothing
end


function equal2!(b_gpu::CuArrays.CuArray{ComplexGPU, 2},
                 i::Int64,
                 a_gpu::CuArrays.CuArray{ComplexGPU, 1})

    function kernel(b, i, a)
        j = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
        if j <= length(a)
            b[i, j] = a[j]
        end
        return nothing
    end

    dev = CUDAnative.CuDevice(0)
    N = length(b_gpu)
    MAX_THREADS = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    threads = min(N, MAX_THREADS)
    blocks = Int(ceil(N / threads))

    @CUDAnative.cuda (blocks, threads) kernel(b_gpu, i, a_gpu)
end


end
