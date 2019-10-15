module Models

import CuArrays
import CUDAdrv
import CUDAnative
import FFTW
using TimerOutputs

import Equations
import Fields
import Fourier
import Grids
import Guards
import Hankel
import LinearPropagators
import Media
import PlasmaEquations
import Units

import PyCall
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


struct ModelR <: Model
    LP :: LinearPropagators.LinearPropagator
    prob :: Equations.Problem
    responses :: Tuple
    keys :: NamedTuple
end


struct ModelT <: Model
    LP :: LinearPropagators.LinearPropagator
    prob :: Equations.Problem
    responses :: Tuple
    PE :: PlasmaEquations.PlasmaEquation
    keys :: NamedTuple
end


struct ModelRT <: Model
    LP :: LinearPropagators.LinearPropagator
    prob :: Equations.Problem
    responses :: Tuple
    PE :: PlasmaEquations.PlasmaEquation
    keys :: NamedTuple
end


struct ModelXY <: Model
    LP :: LinearPropagators.LinearPropagator
    prob :: Equations.Problem
    responses :: Tuple
    keys :: NamedTuple
end


function Model(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.FieldR,
    medium::Media.Medium,
    guard::Guards.GuardR,
    keys::NamedTuple,
    responses_list,
)
    # Linear propagator --------------------------------------------------------
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, guard, field.w0, keys.KPARAXIAL,
    )

    # Nonlinear propagator -----------------------------------------------------
    beta = Media.beta_func(medium, field.w0)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = MU0 * mu * field.w0^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nr)
    if keys.QPARAXIAL
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
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = CuArrays.zeros(ComplexGPU, grid.Nr)

    # Problem:
    p = (responses, Ftmp, guard, keys.QPARAXIAL, QZ, grid.HT)
    pfunc = Equations.PFunction(func_field!, p)
    prob = Equations.Problem(keys.ALG, Ftmp, pfunc)

    # Keys:
    keys = (NONLINEARITY=keys.NONLINEARITY, )

    return ModelR(LP, prob, responses, keys)
end


function Model(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.FieldT,
    medium::Media.Medium,
    guard::Guards.GuardT,
    keys::NamedTuple,
    responses_list,
    plasma_equation::Dict,
)
    # Linear propagator --------------------------------------------------------
    LP = LinearPropagators.LinearPropagator(unit, grid, medium, guard, field.w0)

    # Nonlinear propagator -----------------------------------------------------
    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. MU0 * mu * (grid.w * unit.w)^2 / 2 * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nw)
    for i=1:grid.Nw
        if beta[i] != 0
            QZ[i] = Qfactor[i] / beta[i]
        end
    end
    @. QZ = conj(QZ)

    # Nonlinear responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = zeros(Float64, grid.Nt)
    Etmp = zeros(ComplexF64, grid.Nt)
    Stmp = zeros(ComplexF64, grid.Nw)

    # Problem:
    p = (responses, grid.FT, Etmp, Ftmp, Stmp, guard, QZ)
    pfunc = Equations.PFunction(func_t!, p)
    prob = Equations.Problem(keys.ALG, Stmp, pfunc)

    # Plasma equation ----------------------------------------------------------
    PE = PlasmaEquations.PlasmaEquation(unit, n0, field.w0, plasma_equation)
    if keys.NONLINEARITY
        PE.solve!(field.rho, field.Kdrho, grid.t, field.E)
    end

    # Keys ---------------------------------------------------------------------
    keys = (NONLINEARITY=keys.NONLINEARITY, )

    return ModelT(LP, prob, responses, PE, keys)
end


function Model(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
    medium::Media.Medium,
    guard::Guards.GuardRT,
    keys::NamedTuple,
    responses_list,
    plasma_equation::Dict,
)
    # Linear propagator --------------------------------------------------------
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, guard, field.w0, keys.KPARAXIAL,
    )

    # Nonlinear propagator -----------------------------------------------------
    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. MU0 * mu * (grid.w * unit.w)^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
    if keys.QPARAXIAL
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
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = CuArrays.zeros(FloatGPU, (grid.Nr, grid.Nt))
    Etmp = CuArrays.zeros(ComplexGPU, (grid.Nr, grid.Nt))
    Stmp = CuArrays.zeros(ComplexGPU, (grid.Nr, grid.Nw))

    # Problem:
    p = (
        responses,
        grid.FT,
        Etmp,
        Ftmp,
        Stmp,
        guard,
        keys.QPARAXIAL,
        QZ,
        grid.HT,
    )
    pfunc = Equations.PFunction(func_spectrum!, p)
    prob = Equations.Problem(keys.ALG, Stmp, pfunc)

    # Plasma equation ----------------------------------------------------------
    PE = PlasmaEquations.PlasmaEquation(unit, n0, field.w0, plasma_equation)
    if keys.NONLINEARITY & (! isempty(responses))
        t = range(convert(FloatGPU, grid.tmin),
                  convert(FloatGPU, grid.tmax), length=grid.Nt)
        PE.solve!(field.rho, field.Kdrho, t, field.E)
    end

    # Keys ---------------------------------------------------------------------
    keys = (NONLINEARITY=keys.NONLINEARITY, )

    return ModelRT(LP, prob, responses, PE, keys)
end


function Model(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.FieldXY,
    medium::Media.Medium,
    guard::Guards.GuardXY,
    keys::NamedTuple,
    responses_list,
)
    # Linear propagator --------------------------------------------------------
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, guard, field.w0, keys.KPARAXIAL,
    )

    # Nonlinear propagator -----------------------------------------------------
    beta = Media.beta_func(medium, field.w0)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = MU0 * mu * field.w0^2 / 2. * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if keys.QPARAXIAL
        @. QZ = Qfactor / beta
    else
        for j=1:grid.Ny
        for i=1:grid.Nx
            kzij = sqrt(beta^2 - ((grid.kx[i] * unit.kx)^2 +
                                  (grid.ky[i] * unit.ky)^2) + 0im)
            if kzij != 0.
                QZ[i] = Qfactor / kzij
            end
        end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, QZ))

    # Nonlinear responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = CuArrays.zeros(ComplexGPU, (grid.Nx, grid.Ny))

    # Problem:
    p = (responses, Ftmp, guard, keys.QPARAXIAL, QZ, grid.FT)
    pfunc = Equations.PFunction(func_field!, p)
    prob = Equations.Problem(keys.ALG, Ftmp, pfunc)

    # Keys:
    keys = (NONLINEARITY=keys.NONLINEARITY, )

    return ModelXY(LP, prob, responses, keys)
end


function func_field!(
    dE::CuArrays.CuArray{Complex{T}, 1},
    E::CuArrays.CuArray{Complex{T}, 1},
    z::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    responses, Ftmp, guard, QPARAXIAL, QZ, HT = p

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, z, args)
        Guards.apply_field_filter!(guard, Ftmp)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL
        @. dE = -1im * QZ * dE
    else
        Hankel.dht!(HT, dE)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(guard, dE)
        Hankel.idht!(HT, dE)
    end

    return nothing
end


function func_field!(
    dE::CuArrays.CuArray{Complex{T}, 2},
    E::CuArrays.CuArray{Complex{T}, 2},
    z::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    responses, Ftmp, guard, QPARAXIAL, QZ, FT = p

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, z, args)
        Guards.apply_field_filter!(guard, Ftmp)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL
        @. dE = -1im * QZ * dE
    else
        Fourier.fft!(FT, dE)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(guard, dE)
        Fourier.ifft!(FT, dE)
    end

    return nothing
end


function func_t!(
    dS::AbstractArray{Complex{T}, 1},
    S::AbstractArray{Complex{T}, 1},
    z::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    responses, FT, Etmp, Ftmp, Stmp, guard, QZ = p

    Fourier.hilbert!(FT, S, Etmp)   # spectrum real to signal analytic

    fill!(dS, 0)

    for resp in responses
        resp.calculate(Ftmp, Etmp, z, args)
        Guards.apply_field_filter!(guard, Ftmp)
        Fourier.rfft!(FT, Ftmp, Stmp)   # time -> frequency
        @. dS = dS + resp.Rnl * Stmp
    end

    @. dS = -1im * QZ * dS
    return nothing
end


function func_spectrum!(
    dS::CuArrays.CuArray{Complex{T}, 2},
    S::CuArrays.CuArray{Complex{T}, 2},
    z::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    responses, FT, Etmp, Ftmp, Stmp, guard, QPARAXIAL, QZ, HT = p

    Fourier.hilbert2!(FT, S, Etmp)   # spectrum real to signal analytic

    fill!(dS, 0)

    for resp in responses
        resp.calculate(Ftmp, Etmp, z, args)
        Guards.apply_field_filter!(guard, Ftmp)
        Fourier.rfft2!(FT, Ftmp, Stmp)   # time -> frequency
        update_dS!(dS, resp.Rnl, Stmp)   # dS = dS + Ra * Stmp
    end

    # Nonparaxiality:
    if QPARAXIAL
        @. dS = -1im * QZ * dS
    else
        Hankel.dht!(HT, dS)
        @. dS = -1im * QZ * dS
        Guards.apply_spectral_filter!(guard, dS)
        Hankel.idht!(HT, dS)
    end

    return nothing
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.GridR,
    field::Fields.FieldR,
    guard::Guards.GuardR,
    model::ModelR,
) where T
    z = convert(FloatGPU, z)
    dz = convert(FloatGPU, dz)

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if model.keys.NONLINEARITY & (! isempty(model.responses))
           args = ()
           znew = model.prob.step(field.E, z, dz, args)
           znext = z + dz
           while znew < znext
               dznew = znext - znew
               znew = model.prob.step(field.E, znew, dznew, args)
           end
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        LinearPropagators.propagate!(field.E, model.LP, dz)
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.GridT,
    field::Fields.FieldT,
    guard::Guards.GuardT,
    model::ModelT,
) where T
    # Calculate plasma density -------------------------------------------------
    @timeit "plasma" begin
        if model.keys.NONLINEARITY & (! isempty(model.responses))
            model.PE.solve!(field.rho, field.Kdrho, grid.t, field.E)
        end
    end

    # Field -> temporal spectrum -----------------------------------------------
    @timeit "field -> spectr" begin
        Fourier.rfft!(grid.FT, field.E, field.S)
    end

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if model.keys.NONLINEARITY & (! isempty(model.responses))
           args = ()
           znew = model.prob.step(field.S, z, dz, args)
           znext = z + dz
           while znew < znext
               dznew = znext - znew
               znew = model.prob.step(field.S, znew, dznew, args)
           end
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        LinearPropagators.propagate!(field.S, model.LP, dz)
    end

    # Temporal spectrum -> field -----------------------------------------------
    @timeit "spectr -> field" begin
        Fourier.hilbert!(grid.FT, field.S, field.E)   # spectrum real to signal analytic
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
    end

    return nothing
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
    guard::Guards.GuardRT,
    model::ModelRT,
) where T
    z = convert(FloatGPU, z)
    dz = convert(FloatGPU, dz)

    # Calculate plasma density -------------------------------------------------
    @timeit "plasma" begin
        if model.keys.NONLINEARITY & (! isempty(model.responses))
            t = range(convert(FloatGPU, grid.tmin),
                      convert(FloatGPU, grid.tmax), length=grid.Nt)
            model.PE.solve!(field.rho, field.Kdrho, t, field.E)
            CUDAdrv.synchronize()
        end
    end

    # Field -> temporal spectrum -----------------------------------------------
    @timeit "field -> spectr" begin
        Fourier.rfft2!(grid.FT, field.E, field.S)
        CUDAdrv.synchronize()
    end

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if model.keys.NONLINEARITY & (! isempty(model.responses))
           args = ()
           znew = model.prob.step(field.S, z, dz, args)
           znext = z + dz
           while znew < znext
               dznew = znext - znew
               znew = model.prob.step(field.S, znew, dznew, args)
           end
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        LinearPropagators.propagate!(field.S, model.LP, dz)
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


function zstep(
    z::T,
    dz::T,
    grid::Grids.GridXY,
    field::Fields.FieldXY,
    guard::Guards.GuardXY,
    model::ModelXY,
) where T
    z = convert(FloatGPU, z)
    dz = convert(FloatGPU, dz)

    # Nonlinearity -------------------------------------------------------------
    @timeit "nonlinearity" begin
        if model.keys.NONLINEARITY & (! isempty(model.responses))
           args = ()
           znew = model.prob.step(field.E, z, dz, args)
           znext = z + dz
           while znew < znext
               dznew = znext - znew
               znew = model.prob.step(field.E, znew, dznew, args)
           end
           CUDAdrv.synchronize()
       end
    end

    # Linear propagator --------------------------------------------------------
    @timeit "linear" begin
        LinearPropagators.propagate!(field.E, model.LP, dz)
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
end


function update_dS!(
    dS::CuArrays.CuArray{Complex{T}, 2},
    R::T,
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    @. dS = dS + R * S
    return nothing
end


function update_dS!(
    dS::CuArrays.CuArray{Complex{T}, 2},
    R::CuArrays.CuArray{Complex{T}, 1},
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
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
