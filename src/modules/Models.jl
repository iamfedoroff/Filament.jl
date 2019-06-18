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
import Hankel
import Fourier
import RungeKuttas
import Guards
import NonlinearResponses
import PlasmaEquations

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
    prob :: NamedTuple
    responses :: Tuple
end


struct ModelRT{T} <: Model
    KZ :: CuArrays.CuArray{Complex{T}, 2}
    prob :: NamedTuple
    responses :: Tuple
    PE :: PlasmaEquations.PlasmaEquation
end


struct ModelXY{T} <: Model
    KZ :: CuArrays.CuArray{Complex{T}, 2}
    prob :: NamedTuple
    responses :: Tuple
end


function Model(unit::Units.UnitR, grid::Grids.GridR, field::Fields.FieldR,
               medium::Media.Medium, guard::Guards.GuardR, keys::NamedTuple,
               responses_list)
    # Linear propagator --------------------------------------------------------
    beta = Media.beta_func(medium, field.w0)
    if keys.KPARAXIAL != 0
        KZ = @. beta - (grid.k * unit.k)^2 / (2. * beta)
    else
        KZ = @. sqrt(beta^2 - (grid.k * unit.k)^2 + 0im)
    end
    @. KZ = KZ * unit.z
    @. KZ = conj(KZ)
    KZ = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = MU0 * mu * field.w0^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nr)
    if keys.QPARAXIAL != 0
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
    QZ = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, QZ))

    # Nonlinear responses:
    responses = NonlinearResponses.init(unit, grid, field, medium,
                                        responses_list)

    # Temporary arrays:
    Ftmp = CuArrays.cuzeros(ComplexGPU, grid.Nr)

    # Problem:
    p = (responses, Ftmp, guard, keys.QPARAXIAL, QZ, grid.HT)
    prob = RungeKuttas.Problem(keys.ALG, Ftmp, stepfunc_field!, p)

    return ModelR(KZ, prob, responses)
end


function Model(unit::Units.UnitRT, grid::Grids.GridRT, field::Fields.FieldRT,
               medium::Media.Medium, guard::Guards.GuardRT, keys::NamedTuple,
               responses_list, plasma_equation::Dict)
    # Linear propagator --------------------------------------------------------
    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    KZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
    if keys.KPARAXIAL != 0
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

    KZ = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. MU0 * mu * (grid.w * unit.w)^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
    if keys.QPARAXIAL != 0
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
    QZ = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, QZ))

    # Nonlinear responses:
    responses = NonlinearResponses.init(unit, grid, field, medium,
                                        responses_list)

    # Temporary arrays:
    Ftmp = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    Etmp = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nt))
    Stmp = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nw))

    # Problem:
    p = (responses, grid.FT, Etmp, Ftmp, Stmp, guard, keys.QPARAXIAL, QZ, grid.HT)
    prob = RungeKuttas.Problem(keys.ALG, Stmp, stepfunc_spectrum!, p)

    # Plasma equation ----------------------------------------------------------
    PE = PlasmaEquations.PlasmaEquation(unit, grid, field, medium,
                                        plasma_equation)
    PlasmaEquations.solve!(PE, field.rho, field.Kdrho, field.E)

    return ModelRT(KZ, prob, responses, PE)
end


function Model(unit::Units.UnitXY, grid::Grids.GridXY, field::Fields.FieldXY,
               medium::Media.Medium, guard::Guards.GuardXY, keys::NamedTuple,
               responses_list)
    # Linear propagator --------------------------------------------------------
    beta = Media.beta_func(medium, field.w0)
    KZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if keys.KPARAXIAL != 0
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
    KZ = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = MU0 * mu * field.w0^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if keys.QPARAXIAL != 0
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
    QZ = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, QZ))

    # Nonlinear responses:
    responses = NonlinearResponses.init(unit, grid, field, medium,
                                        responses_list)

    # Temporary arrays:
    Ftmp = CuArrays.cuzeros(ComplexGPU, (grid.Nx, grid.Ny))

    # Problem:
    p = (responses, Ftmp, guard, keys.QPARAXIAL, QZ, grid.FT)
    prob = RungeKuttas.Problem(keys.ALG, Ftmp, stepfunc_field!, p)

    return ModelXY(KZ, prob, responses)
end


function dzadapt(model::Model, phimax::AbstractFloat)
    if ! isempty(model.responses)
        dzs = []
        for resp in model.responses
            dz = NonlinearResponses.dzadaptive(resp, phimax)
            push!(dzs, dz)
        end
        dz = minimum(dzs)
    else
        dz = Inf
    end
    return dz
end


function stepfunc_field!(dE::CuArrays.CuArray{Complex{T}, 1},
                         E::CuArrays.CuArray{Complex{T}, 1},
                         p::Tuple, args::Tuple) where T

    responses, Ftmp, guard, QPARAXIAL, QZ, HT = p
    z, = args

    fill!(dE, 0)

    for resp in responses
        NonlinearResponses.calculate!(resp, z, Ftmp, E)
        Guards.apply_field_filter!(guard, Ftmp)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL != 0
        @. dE = -1im * QZ * dE
    else
        Hankel.dht!(HT, dE)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(guard, dE)
        Hankel.idht!(HT, dE)
    end

    return nothing
end


function stepfunc_field!(dE::CuArrays.CuArray{Complex{T}, 2},
                         E::CuArrays.CuArray{Complex{T}, 2},
                         p::Tuple, args::Tuple) where T
    responses, Ftmp, guard, QPARAXIAL, QZ, FT = p
    z, = args

    fill!(dE, 0)

    for resp in responses
        NonlinearResponses.calculate!(resp, z, Ftmp, E)
        Guards.apply_field_filter!(guard, Ftmp)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL != 0
        @. dE = -1im * QZ * dE
    else
        Fourier.fft!(FT, dE)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(guard, dE)
        Fourier.ifft!(FT, dE)
    end

    return nothing
end


function stepfunc_spectrum!(dS::CuArrays.CuArray{Complex{T}, 2},
                            S::CuArrays.CuArray{Complex{T}, 2},
                            p::Tuple, args::Tuple) where T
    responses, FT, Etmp, Ftmp, Stmp, guard, QPARAXIAL, QZ, HT = p
    z, = args

    Fourier.hilbert2!(FT, S, Etmp)   # spectrum real to signal analytic

    fill!(dS, 0)

    for resp in responses
        NonlinearResponses.calculate!(resp, z, Ftmp, Etmp)
        Guards.apply_field_filter!(guard, Ftmp)
        Fourier.rfft2!(FT, Ftmp, Stmp)   # time -> frequency
        update_dS!(dS, resp.Rnl, Stmp)   # dS = dS + Ra * Stmp
    end

    # Nonparaxiality:
    if QPARAXIAL != 0
        @. dS = -1im * QZ * dS
    else
        Hankel.dht!(HT, dS)
        @. dS = -1im * QZ * dS
        Guards.apply_spectral_filter!(guard, dS)
        Hankel.idht!(HT, dS)
    end

    return nothing
end


function zstep(z::T, dz::T, grid::Grids.GridR, field::Fields.FieldR,
               guard::Guards.GuardR, model::ModelR) where T
    z_gpu = FloatGPU(z)
    dz_gpu = FloatGPU(dz)

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if ! isempty(model.responses)
           args = (z_gpu, )
           RungeKuttas.step(model.prob, field.E, dz_gpu, args)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Hankel.dht!(grid.HT, field.E)
        @. field.E = field.E * CUDAnative.exp(-1im * model.KZ * dz_gpu)
        Guards.apply_spectral_filter!(guard, field.E)
        Hankel.idht!(grid.HT, field.E)
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
end


function zstep(z::T, dz::T, grid::Grids.GridRT, field::Fields.FieldRT,
               guard::Guards.GuardRT, model::ModelRT) where T
    # Calculate plasma density -------------------------------------------------
    @timeit "plasma" begin
        if ! isempty(model.PE.probs)
            PlasmaEquations.solve!(model.PE, field.rho, field.Kdrho, field.E)
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
           args = (z_gpu, )
           RungeKuttas.step(model.prob, field.S, dz_gpu, args)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Hankel.dht!(grid.HT, field.S)
        @. field.S = field.S * CUDAnative.exp(-1im * model.KZ * dz_gpu)
        Guards.apply_spectral_filter!(guard, field.S)
        Hankel.idht!(grid.HT, field.S)
        CUDAdrv.synchronize()
    end

    # Temporal spectrum -> field -----------------------------------------------
    @timeit "spectr -> field" begin
        Fourier.hilbert2!(grid.FT, field.S, field.E)   # spectrum real to signal analytic
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
end


function zstep(z::T, dz::T, grid::Grids.GridXY, field::Fields.FieldXY,
               guard::Guards.GuardXY, model::ModelXY) where T
    z_gpu = FloatGPU(z)
    dz_gpu = FloatGPU(dz)

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if ! isempty(model.responses)
           args = (z_gpu, )
           RungeKuttas.step(model.prob, field.E, dz_gpu, args)
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        Fourier.fft!(grid.FT, field.E)
        @. field.E = field.E * CUDAnative.exp(-1im * model.KZ * dz_gpu)
        Guards.apply_spectral_filter!(guard, field.E)
        Fourier.ifft!(grid.FT, field.E)
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
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
