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
import NonlinearResponses

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = scipy_constants.mu_0   # the magnetic constant [N/A^2]
const QE = scipy_constants.e   # elementary charge [C]
const ME = scipy_constants.m_e   # electron mass [kg]
const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32
const ComplexGPU = ComplexF32


abstract type Model end


struct ModelR{T} <: Model
    KZ :: CuArrays.CuArray{Complex{T}, 1}
    QZ :: CuArrays.CuArray{Complex{T}, 1}
    phi_kerr :: Float64
    guard :: Guards.Guard
    RK :: RungeKuttas.RungeKutta
    keys :: Dict

    Ftmp :: CuArrays.CuArray{Complex{T}, 1}

    responses #:: Array{NonlinearResponses.NonlinearResponse, 1}
end


struct ModelRT{T} <: Model
    KZ :: CuArrays.CuArray{Complex{T}, 2}
    QZ :: CuArrays.CuArray{Complex{T}, 2}
    phi_kerr :: Float64
    phi_plasma :: Float64
    guard :: Guards.Guard
    RK :: RungeKuttas.RungeKutta
    keys :: Dict

    Ftmp :: CuArrays.CuArray{T, 2}
    Etmp :: CuArrays.CuArray{Complex{T}, 2}
    Stmp :: CuArrays.CuArray{Complex{T}, 2}

    responses #:: Array{NonlinearResponses.NonlinearResponse, 1}
end


struct ModelXY{T} <: Model
    KZ :: CuArrays.CuArray{Complex{T}, 2}
    QZ :: CuArrays.CuArray{Complex{T}, 2}
    phi_kerr :: Float64
    guard :: Guards.Guard
    RK :: RungeKuttas.RungeKutta
    keys :: Dict

    Ftmp :: CuArrays.CuArray{Complex{T}, 2}

    responses #:: Array{NonlinearResponses.NonlinearResponse, 1}
end


function Model(unit::Units.UnitR, grid::Grids.GridR, field::Fields.FieldR,
               medium::Media.Medium, keys::Dict, p_guard::Tuple, dict_responses)
    # Guards -------------------------------------------------------------------
    guard = Guards.Guard(unit, grid, field.w0, medium, p_guard...)

    # Runge-Kutta --------------------------------------------------------------
    RKORDER = keys["RKORDER"]
    RK = RungeKuttas.RungeKutta(RKORDER, ComplexGPU, grid.Nr)

    # Linear propagator --------------------------------------------------------
    KPARAXIAL = keys["KPARAXIAL"]

    beta = Media.beta_func(medium, field.w0)
    if KPARAXIAL != 0
        KZ = @. beta - (grid.k * unit.k)^2 / (2. * beta)
    else
        KZ = @. sqrt(beta^2 - (grid.k * unit.k)^2 + 0im)
    end
    @. KZ = KZ * unit.z
    @. KZ = conj(KZ)
    KZ = CuArrays.cu(convert(Array{ComplexGPU, 1}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    QPARAXIAL = keys["QPARAXIAL"]

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = MU0 * mu * field.w0^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nr)
    if QPARAXIAL != 0
        @. QZ = Qfactor / beta
    else
        for i=1:grid.Nr
            kzi = sqrt(beta^2 - (grid.k[i] * unit.k)^2 + 0im)
            if kzi != 0.
                QZ[i] = Qfactor / kzi
            end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.cu(convert(Array{ComplexGPU, 1}, QZ))

    # Nonlinear responses ------------------------------------------------------
    phi_kerr = phi_kerr_func(unit, field, medium)

    responses = []
    # responses = Array{NonlinearResponses.NonlinearResponse}(undef, 1)
    for dict_response in dict_responses
        init = dict_response["init"]
        Rnl, calc, p = init(unit, grid, field, medium, 0, dict_response)
        response = NonlinearResponses.NonlinearResponse(Rnl, calc, p)
        push!(responses, response)
    end

    # Temporary arrays ---------------------------------------------------------
    Ftmp = CuArrays.cuzeros(ComplexGPU, grid.Nr)

    return ModelR(KZ, QZ, phi_kerr, guard, RK, keys, Ftmp, responses)
end


function Model(unit::Units.UnitRT, grid::Grids.GridRT, field::Fields.FieldRT,
               medium::Media.Medium, plasma::Plasmas.Plasma, keys::Dict,
               p_guard::Tuple, dict_responses)
    # Guards -------------------------------------------------------------------
    guard = Guards.Guard(unit, grid, medium, p_guard...)

    # Runge-Kutta --------------------------------------------------------------
    RKORDER = keys["RKORDER"]
    RK = RungeKuttas.RungeKutta(RKORDER, ComplexGPU, grid.Nr, grid.Nw)

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

    # Nonlinear responses ------------------------------------------------------
    phi_kerr = phi_kerr_func(unit, field, medium)
    phi_plasma = phi_plasma_func(unit, field, medium, plasma)

    responses = []
    # responses = Array{NonlinearResponses.NonlinearResponse}(undef, 1)
    for dict_response in dict_responses
        init = dict_response["init"]
        Rnl, calc, p = init(unit, grid, field, medium, plasma, dict_response)
        response = NonlinearResponses.NonlinearResponse(Rnl, calc, p)
        push!(responses, response)
    end

    # Temporary arrays ---------------------------------------------------------
    Ftmp = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    Etmp = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nt))
    Stmp = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nw))

    return ModelRT(KZ, QZ, phi_kerr, phi_plasma, guard,
                   RK, keys, Ftmp, Etmp, Stmp, responses)
end


function Model(unit::Units.UnitXY, grid::Grids.GridXY, field::Fields.FieldXY,
               medium::Media.Medium, keys::Dict, p_guard::Tuple, dict_responses)
    # Guards -------------------------------------------------------------------
    guard = Guards.Guard(unit, grid, field.w0, medium, p_guard...)

    # Runge-Kutta --------------------------------------------------------------
    RKORDER = keys["RKORDER"]
    RK = RungeKuttas.RungeKutta(RKORDER, ComplexGPU, grid.Nx, grid.Ny)

    # Linear propagator --------------------------------------------------------
    KPARAXIAL = keys["KPARAXIAL"]

    beta = Media.beta_func(medium, field.w0)
    KZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if KPARAXIAL != 0
        for j=1:grid.Ny
            for i=1:grid.Nx
                KZ[i, j] = beta - ((grid.kx[i] * unit.kx)^2 + (grid.ky[j] * unit.ky)^2) / (2. * beta)
            end
        end
    else
        for j=1:grid.Ny
            for i=1:grid.Nx
                KZ[i, j] = sqrt(beta^2 - ((grid.kx[i] * unit.kx)^2 + (grid.ky[j] * unit.ky)^2) + 0im)
            end
        end
    end
    @. KZ = KZ * unit.z
    @. KZ = conj(KZ)
    KZ = CuArrays.cu(convert(Array{ComplexGPU, 2}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    QPARAXIAL = keys["QPARAXIAL"]

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = MU0 * mu * field.w0^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if QPARAXIAL != 0
        @. QZ = Qfactor / beta
    else
        for j=1:grid.Ny
            for i=1:grid.Nx
                kzij = sqrt(beta^2 - ((grid.kx[i] * unit.kx)^2 + (grid.ky[i] * unit.ky)^2) + 0im)
                if kzij != 0.
                    QZ[i] = Qfactor / kzij
                end
            end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.cu(convert(Array{ComplexGPU, 2}, QZ))

    # Nonlinear responses ------------------------------------------------------
    phi_kerr = phi_kerr_func(unit, field, medium)

    responses = []
    # responses = Array{NonlinearResponses.NonlinearResponse}(undef, 1)
    for dict_response in dict_responses
        init = dict_response["init"]
        Rnl, calc, p = init(unit, grid, field, medium, 0, dict_response)
        response = NonlinearResponses.NonlinearResponse(Rnl, calc, p)
        push!(responses, response)
    end

    # Temporary arrays ---------------------------------------------------------
    Ftmp = CuArrays.cuzeros(ComplexGPU, (grid.Nx, grid.Ny))

    return ModelXY(KZ, QZ, phi_kerr, guard, RK, keys, Ftmp, responses)
end


function adaptive_dz(model::Model, AdaptLevel::Float64, I::Float64)
    if ! isempty(model.responses)
        dz_kerr = model.phi_kerr / I * AdaptLevel
    else
        dz_kerr = Inf
    end
    return dz_kerr
end


function adaptive_dz(model::ModelRT, AdaptLevel::Float64, I::Float64,
                     rho::Float64)
    if ! isempty(model.responses)
        dz_kerr = model.phi_kerr / I * AdaptLevel
    else
        dz_kerr = Inf
    end

    if (! isempty(model.responses)) & (rho != 0.)
        dz_plasma = model.phi_plasma / rho * AdaptLevel
    else
        dz_plasma = Inf
    end

    dz = min(dz_kerr, dz_plasma)
    return dz
end


function rkfunc_field!(dE::CuArrays.CuArray{Complex{T}, 1},
                       E::CuArrays.CuArray{Complex{T}, 1},
                       p::Tuple) where T
    z = p[1]
    grid = p[2]
    model = p[3]

    fill!(dE, 0)

    for response in model.responses
        NonlinearResponses.calculate!(response, z, model.Ftmp, E)
        Guards.apply_spatial_filter!(model.guard, model.Ftmp)
        @. dE = dE + response.Rnl * model.Ftmp
    end

    # Nonparaxiality:
    if model.keys["QPARAXIAL"] != 0
        @. dE = -1im * model.QZ * dE
    else
        Hankel.dht!(grid.HT, dE)
        @. dE = -1im * model.QZ * dE
        Guards.apply_angular_filter!(model.guard, dE)
        Hankel.idht!(grid.HT, dE)
    end

    return nothing
end


function rkfunc_field!(dE::CuArrays.CuArray{Complex{T}, 2},
                       E::CuArrays.CuArray{Complex{T}, 2},
                       p::Tuple) where T
    z = p[1]
    grid = p[2]
    model = p[3]

    fill!(dE, 0)

    for response in model.responses
        NonlinearResponses.calculate!(response, z, model.Ftmp, E)
        Guards.apply_spatial_filter!(model.guard, model.Ftmp)
        @. dE = dE + response.Rnl * model.Ftmp
    end

    # Nonparaxiality:
    if model.keys["QPARAXIAL"] != 0
        @. dE = -1im * model.QZ * dE
    else
        Fourier.fft!(grid.FT, dE)
        @. dE = -1im * model.QZ * dE
        Guards.apply_angular_filter!(model.guard, dE)
        Fourier.ifft!(grid.FT, dE)
    end

    return nothing
end


function rkfunc_spectrum!(dS::CuArrays.CuArray{Complex{T}, 2},
                          S::CuArrays.CuArray{Complex{T}, 2},
                          p::Tuple) where T
    z = p[1]
    grid = p[2]
    model = p[3]

    Fourier.hilbert2!(grid.FT, S, model.Etmp)   # spectrum real to signal analytic

    fill!(dS, 0)

    for response in model.responses
        NonlinearResponses.calculate!(response, z, model.Ftmp, model.Etmp)
        Guards.apply_spatio_temporal_filter!(model.guard, model.Ftmp)
        Fourier.rfft2!(grid.FT, model.Ftmp, model.Stmp)   # time -> frequency
        update_dS!(dS, response.Rnl, model.Stmp)   # dS = dS + Ra * Stmp
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


function zstep(z::Float64, dz::Float64, grid::Grids.GridR, field::Fields.FieldR,
               model::ModelR)
    z_gpu = FloatGPU(z)
    dz_gpu = FloatGPU(dz)

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if ! isempty(model.responses)
           p = (z_gpu, grid, model)
           RungeKuttas.solve!(model.RK, field.E, dz_gpu, rkfunc_field!, p)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Hankel.dht!(grid.HT, field.E)
        @. field.E = field.E * CUDAnative.exp(-1im * model.KZ * dz_gpu)
        Guards.apply_angular_filter!(model.guard, field.E)
        Hankel.idht!(grid.HT, field.E)
        CUDAdrv.synchronize()
    end

    @timeit "spatial filter" begin
        Guards.apply_spatial_filter!(model.guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
end


function zstep(z::Float64, dz::Float64, grid::Grids.GridRT,
               field::Fields.FieldRT, plasma::Plasmas.Plasma, model::ModelRT)
    # Calculate plasma density -------------------------------------------------
    @timeit "plasma" begin
        if ! isempty(plasma.components)
            Plasmas.free_charge(plasma, grid, field)
            CUDAdrv.synchronize()
        end
    end

    z_gpu = FloatGPU(z)
    dz_gpu = FloatGPU(dz)

    # Field -> temporal spectrum -----------------------------------------------
    @timeit "field -> spectr" begin
        Fourier.rfft2!(grid.FT, field.E, field.S)
        CUDAdrv.synchronize()
    end

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if ! isempty(model.responses)
           p = (z_gpu, grid, model)
           RungeKuttas.solve!(model.RK, field.S, dz_gpu, rkfunc_spectrum!, p)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Hankel.dht!(grid.HT, field.S)
        @. field.S = field.S * CUDAnative.exp(-1im * model.KZ * dz_gpu)
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


function zstep(z::Float64, dz::Float64, grid::Grids.GridXY,
               field::Fields.FieldXY, model::ModelXY)
    z_gpu = FloatGPU(z)
    dz_gpu = FloatGPU(dz)

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if ! isempty(model.responses)
           p = (z_gpu, grid, model)
           RungeKuttas.solve!(model.RK, field.E, dz_gpu, rkfunc_field!, p)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Fourier.fft!(grid.FT, field.E)
        @. field.E = field.E * CUDAnative.exp(-1im * model.KZ * dz_gpu)
        Guards.apply_angular_filter!(model.guard, field.E)
        Fourier.ifft!(grid.FT, field.E)
        CUDAdrv.synchronize()
    end

    @timeit "spatial filter" begin
        Guards.apply_spatial_filter!(model.guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
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
function phi_plasma_func(unit::Units.Unit, field::Fields.Field,
                         medium::Media.Medium, plasma::Plasmas.Plasma)
    w0 = field.w0
    k0 = Media.k_func(medium, w0)
    nuc = plasma.nuc
    mu = medium.permeability(w0)
    MR = plasma.mr * ME   # reduced mass of electron and hole (effective mass)
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


function update_dS!(dS::CuArrays.CuArray{Complex{T}, 2},
                    R::T,
                    S::CuArrays.CuArray{Complex{T}, 2}) where T
    @. dS = dS + R * S
    return nothing
end


function update_dS!(dS::CuArrays.CuArray{Complex{T}, 2},
                    R::CuArrays.CuArray{Complex{T}, 1},
                    S::CuArrays.CuArray{Complex{T}, 2}) where T
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


"""
Complex version of CUDAnative.exp function. Adapted from
https://discourse.julialang.org/t/base-function-in-cuda-kernels/21866/4
"""
@inline function CUDAnative.exp(x::Complex{T}) where T
    scale = CUDAnative.exp(x.re)
    return Complex{T}(scale * CUDAnative.cos(x.im), scale * CUDAnative.sin(x.im))
end


end
