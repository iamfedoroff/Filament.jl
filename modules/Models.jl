module Models

using TimerOutputs
import CuArrays
import CUDAnative
import CUDAdrv
import FFTW

import PyCall

import Units
import Grids
import Fields
import Media
import Plasmas
import Hankel
import Fourier
import RungeKuttas
import Guards

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = scipy_constants.mu_0   # the magnetic constant [N/A^2]
const QE = scipy_constants.e   # elementary charge [C]
const ME = scipy_constants.m_e   # electron mass [kg]
const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32
const ComplexGPU = ComplexF32


struct Model
    KZ :: CuArrays.CuArray{ComplexGPU, 2}
    QZ :: CuArrays.CuArray{ComplexGPU, 2}
    Rk :: FloatGPU
    Rr :: FloatGPU
    Hramanw :: CuArrays.CuArray{ComplexGPU, 1}
    Rp :: CuArrays.CuArray{ComplexGPU, 1}
    Ra :: CuArrays.CuArray{ComplexGPU, 1}
    phi_kerr :: Float64
    phi_plasma :: Float64
    guard :: Guards.GuardFilter
    RK :: Union{RungeKuttas.RungeKutta2, RungeKuttas.RungeKutta3,
                RungeKuttas.RungeKutta4}
    keys :: Dict

    Ftmp :: CuArrays.CuArray{FloatGPU, 2}
    Etmp :: CuArrays.CuArray{ComplexGPU, 2}
    Stmp :: CuArrays.CuArray{ComplexGPU, 2}
end


function Model(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
               medium::Media.Medium, plasma::Plasmas.Plasma, keys::Dict)
    # Guards -------------------------------------------------------------------
    rguard_width = keys["rguard_width"]
    tguard_width = keys["tguard_width"]
    kguard = keys["kguard"]
    wguard = keys["wguard"]
    guard = Guards.GuardFilter(unit, grid, medium,
                               rguard_width, tguard_width, kguard, wguard)

    # Runge-Kutta --------------------------------------------------------------
    RKORDER = keys["RKORDER"]
    RK = RungeKuttas.RungeKutta(RKORDER, grid.Nr, grid.Nw)

    # Linear propagator --------------------------------------------------------
    KPARAXIAL = keys["KPARAXIAL"]

    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    KZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
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

    KZ = CuArrays.cu(convert(Array{ComplexGPU, 2}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    QPARAXIAL = keys["QPARAXIAL"]

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. MU0 * mu * (grid.w * unit.w)^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
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

    QZ = CuArrays.cu(convert(Array{ComplexGPU, 2}, QZ))

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

    Rk = FloatGPU(Rk)
    Rr = FloatGPU(Rr)

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put H(t)
    # into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. medium.raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * grid.dt * unit.t

    if abs(1. - sum(Hraman)) > 1e-3
        println("WARNING: The integral of Raman response function should be" *
                " normalized to 1.")
    end

    Tguard = convert(Array{ComplexF64, 1}, CuArrays.collect(guard.T))
    @. Hraman = Hraman * Tguard   # temporal filter
    Hraman = Fourier.ifftshift(Hraman)
    Hramanw = FFTW.rfft(Hraman)   # time -> frequency

    Hramanw = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Hramanw))

    # Plasma nonlinearity ------------------------------------------------------
    Rp = Rp_func(unit, grid, field, medium, plasma)
    @. Rp = conj(Rp)
    phi_plasma = phi_kerr_func(unit, field, medium)

    Rp = CuArrays.cu(convert(Array{ComplexGPU, 1}, Rp))

    # Losses due to multiphoton ionization -------------------------------------
    Ra = Ra_func(unit, grid, field, medium)
    @. Ra = conj(Ra)

    Ra = CuArrays.cu(convert(Array{ComplexGPU, 1}, Ra))

    # Temporary arrays:
    Ftmp = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    Etmp = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nt))
    Stmp = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nw))

    return Model(KZ, QZ, Rk, Rr, Hramanw, Rp, Ra, phi_kerr, phi_plasma, guard,
                 RK, keys, Ftmp, Etmp, Stmp)
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


function func!(dS::CuArrays.CuArray{ComplexGPU, 2},
               S::CuArrays.CuArray{ComplexGPU, 2},
               p::Tuple)
    grid = p[1]
    model = p[2]
    plasma = p[3]

    fill!(dS, FloatGPU(0.))

    Fourier.hilbert2!(grid.FT, S, model.Etmp)   # spectrum real to signal analytic

    # Kerr nonlinearity:
    if model.keys["KERR"] != 0
        if model.keys["THG"] != 0
            @. model.Ftmp = real(model.Etmp)^3
        else
            @. model.Ftmp = FloatGPU(3. / 4.) * abs2(model.Etmp) * real(model.Etmp)
        end
        Guards.apply_spatio_temporal_filter!(model.guard, model.Ftmp)
        Fourier.rfft2!(grid.FT, model.Ftmp, model.Stmp)   # time -> frequency
        @. dS = dS + model.Rk * model.Stmp
    end

    # Stimulated Raman nonlinearity:
    if model.keys["RAMAN"] != 0
        if model.keys["RTHG"] != 0
            @. model.Ftmp = real(model.Etmp)^2
        else
            @. model.Ftmp = FloatGPU(3. / 4.) * abs2(model.Etmp)
        end
        Fourier.convolution2!(grid.FT, model.Hramanw, model.Ftmp)
        @. model.Ftmp = model.Ftmp * real(model.Etmp)
        Guards.apply_spatio_temporal_filter!(model.guard, model.Ftmp)
        Fourier.rfft2!(grid.FT, model.Ftmp, model.Stmp)   # time -> frequency
        @. dS = dS + model.Rr * model.Stmp
    end

    # Plasma nonlinearity:
    if model.keys["PLASMA"] != 0
        @. model.Ftmp = plasma.rho * real(model.Etmp)
        Guards.apply_spatio_temporal_filter!(model.guard, model.Ftmp)
        Fourier.rfft2!(grid.FT, model.Ftmp, model.Stmp)   # time -> frequency
        update_dS!(dS, model.Rp, model.Stmp)   # dS = dS + Rp * Stmp
    end

    # Losses due to multiphoton ionization:
    if model.keys["ILOSSES"] != 0
        if model.keys["IONARG"] != 0
            @. model.Ftmp = abs2(model.Etmp)
        else
            @. model.Ftmp = real(model.Etmp)^2
        end
        inverse!(model.Ftmp)
        @. model.Ftmp = plasma.Kdrho * model.Ftmp * real(model.Etmp)
        Guards.apply_spatio_temporal_filter!(model.guard, model.Ftmp)
        Fourier.rfft2!(grid.FT, model.Ftmp, model.Stmp)   # time -> frequency
        update_dS!(dS, model.Ra, model.Stmp)   # dS = dS + Ra * Stmp
    end

    # Nonparaxiality:
    if model.keys["QPARAXIAL"] != 0
        @. dS = -1im * model.QZ * dS
    else
        Hankel.dht!(grid.HT, dS)
        @. dS = -1im * model.QZ * dS
        Guards.apply_frequency_angular_filter!(model.guard, dS)
        Hankel.idht!(grid.HT, dS)
    end

    return nothing
end


function zstep(dz::Float64, grid::Grids.Grid, field::Fields.Field,
               plasma::Plasmas.Plasma, model::Model)
    # Calculate plasma density -------------------------------------------------
    @timeit "plasma" begin
        if (model.keys["PLASMA"] != 0) | (model.keys["ILOSSES"] != 0)
            Plasmas.free_charge(plasma, grid, field)
            CUDAdrv.synchronize()
        end
    end

    dz_gpu = FloatGPU(dz)

    # Field -> temporal spectrum -----------------------------------------------
    @timeit "field -> spectr" begin
        Fourier.rfft2!(grid.FT, field.E, field.S)
        CUDAdrv.synchronize()
    end

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if (model.keys["KERR"] != 0) | (model.keys["PLASMA"] != 0) |
           (model.keys["ILOSSES"] != 0)
           p = (grid, model, plasma)
           RungeKuttas.solve!(model.RK, field.S, dz_gpu, func!, p)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Hankel.dht!(grid.HT, field.S)
        linear_propagator!(field.S, model.KZ, dz_gpu)   # S = S * exp(KZ * dz)
        Guards.apply_frequency_angular_filter!(model.guard, field.S)
        Hankel.idht!(grid.HT, field.S)
        CUDAdrv.synchronize()
    end

    # Temporal spectrum -> field -----------------------------------------------
    @timeit "spectr -> field" begin
        Fourier.hilbert2!(grid.FT, field.S, field.E)   # spectrum real to signal analytic
        CUDAdrv.synchronize()
    end

    @timeit "sp-temp filter" begin
        Guards.apply_spatio_temporal_filter!(model.guard, field.E)
        CUDAdrv.synchronize()
    end

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
    R = zeros(ComplexF64, grid.Nw)
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

    R = zeros(ComplexF64, grid.Nw)
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


function linear_propagator!(S, KZ, dz)
    N1, N2 = size(S)
    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nth = min(N1 * N2, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N1 * N2 / nth))
    @CUDAnative.cuda blocks=nbl threads=nth linear_propagator_kernel(S, KZ, dz)
end


function linear_propagator_kernel(S, KZ, dz)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(S)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        # Unfortunately, "CUDAnative.exp(x)" function does not work with
        # complex arguments. To solve the issue, I use Euler's formula:
        #     exp(-1im * x) = (cos(xr) - 1im * sin(xr)) * exp(xi),
        # where xr = real(x) and xi = imag(x).
        xr = real(KZ[i, j]) * dz
        xi = imag(KZ[i, j]) * dz
        expval = (CUDAnative.cos(xr) - ComplexGPU(1im) * CUDAnative.sin(xr)) *
                 CUDAnative.exp(xi)
        S[i, j] = S[i, j] * expval
    end
    return nothing
end


function inverse!(F)
    N1, N2 = size(F)
    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nth = min(N1 * N2, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N1 * N2 / nth))
    @CUDAnative.cuda blocks=nbl threads=nth inverse_kernel(F)
    return nothing
end

function inverse_kernel(F)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(F)
    for k=id:stride:N1*N2
        if F[k] >= FloatGPU(1e-30)
            F[k] = FloatGPU(1.) / F[k]
        else
            F[k] = FloatGPU(0.)
        end
    end
    return nothing
end

function update_dS!(dS, R, S)
    N1, N2 = size(S)
    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nth = min(N1 * N2, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N1 * N2 / nth))
    @CUDAnative.cuda blocks=nbl threads=nth update_dS_kernel(dS, R, S)
    return nothing
end


function update_dS_kernel(dS, R, S)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nw = size(S)
    cartesian = CartesianIndices((Nr, Nw))
    for k=id:stride:Nr*Nw
        i = cartesian[k][1]
        j = cartesian[k][2]
        dS[i, j] = dS[i, j] + R[j] * S[i, j]
    end
    return nothing
end


end
