module NonlinearPropagators

import CuArrays
import CUDAdrv
import CUDAnative

import Equations
import Fields
import Fourier
import Grids
import Guards
import Media
import Units

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const MU0 = scipy_constants.mu_0   # the magnetic constant [N/A^2]

const FloatGPU = Float32
const MAX_THREADS_PER_BLOCK =
        CUDAdrv.attribute(
            CUDAnative.CuDevice(0),
            CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )


struct NonlinearPropagator
    prob :: Equations.Problem
end


function NonlinearPropagator(
    unit::Units.UnitR,
    grid::Grids.GridR,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    responses_list,
    keys::NamedTuple,
)
    QPARAXIAL = keys.QPARAXIAL
    ALG = keys.ALG

    # Prefactor:
    beta = Media.beta_func(medium, field.w0)
    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = 0.5 * MU0 * mu * field.w0^2 * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nr)
    if QPARAXIAL
        @. QZ = Qfactor / beta
    else
        for ir=1:grid.Nr
            kzi = sqrt(beta^2 - (grid.k[ir] * unit.k)^2 + 0im)
            if kzi != 0
                QZ[ir] = Qfactor / kzi
            end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.CuArray(convert(Array{Complex{FloatGPU}, 1}, QZ))

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = CuArrays.zeros(Complex{FloatGPU}, grid.Nr)

    # Problem:
    p = (responses, Ftmp, guard, QPARAXIAL, QZ, grid.HT)
    pfunc = Equations.PFunction(_func_r!, p)
    prob = Equations.Problem(ALG, Ftmp, pfunc)

    return NonlinearPropagator(prob)
end


function NonlinearPropagator(
    unit::Units.UnitT,
    grid::Grids.GridT,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    responses_list,
    keys::NamedTuple,
)
    ALG = keys.ALG

    # Prefactor:
    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. 0.5 * MU0 * mu * (grid.w * unit.w)^2 * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nw)
    for iw=1:grid.Nw
        if beta[iw] != 0
            QZ[iw] = Qfactor[iw] / beta[iw]
        end
    end
    @. QZ = conj(QZ)

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = zeros(grid.Nt)
    Etmp = zeros(ComplexF64, grid.Nt)
    Stmp = zeros(ComplexF64, grid.Nw)

    # Problem:
    p = (responses, grid.FT, Etmp, Ftmp, Stmp, guard, QZ)
    pfunc = Equations.PFunction(_func_t!, p)
    prob = Equations.Problem(ALG, Stmp, pfunc)

    return NonlinearPropagator(prob)
end


function NonlinearPropagator(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    responses_list,
    keys::NamedTuple,
)
    QPARAXIAL = keys.QPARAXIAL
    ALG = keys.ALG

    # Prefactor:
    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. 0.5 * MU0 * mu * (grid.w * unit.w)^2 * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
    if QPARAXIAL
        for iw=1:grid.Nw
            if beta[iw] != 0
                for ir=1:grid.Nr
                    QZ[ir, iw] = Qfactor[iw] / beta[iw]
                end
            end
        end
    else
        for iw=1:grid.Nw
        for ir=1:grid.Nr
            kzij = sqrt(beta[iw]^2 - (grid.k[ir] * unit.k)^2 + 0im)
            if kzij != 0
                QZ[ir, iw] = Qfactor[iw] / kzij
            end
        end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.CuArray(convert(Array{Complex{FloatGPU}, 2}, QZ))

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = CuArrays.zeros(FloatGPU, (grid.Nr, grid.Nt))
    Etmp = CuArrays.zeros(Complex{FloatGPU}, (grid.Nr, grid.Nt))
    Stmp = CuArrays.zeros(Complex{FloatGPU}, (grid.Nr, grid.Nw))

    # Problem:
    p = (responses, grid.FT, Etmp, Ftmp, Stmp, guard, QPARAXIAL, QZ, grid.HT)
    pfunc = Equations.PFunction(_func_rt!, p)
    prob = Equations.Problem(ALG, Stmp, pfunc)

    return NonlinearPropagator(prob)
end


function NonlinearPropagator(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    responses_list,
    keys::NamedTuple,
)
    QPARAXIAL = keys.QPARAXIAL
    ALG = keys.ALG

    # Prefactor:
    beta = Media.beta_func(medium, field.w0)
    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, n0)
    mu = medium.permeability(field.w0)

    Qfactor = 0.5 * MU0 * mu * field.w0^2 * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if QPARAXIAL
        @. QZ = Qfactor / beta
    else
        for iy=1:grid.Ny
        for ix=1:grid.Nx
            kzij = sqrt(beta^2 - ((grid.kx[ix] * unit.kx)^2 +
                                  (grid.ky[iy] * unit.ky)^2) + 0im)
            if kzij != 0
                QZ[ix, iy] = Qfactor / kzij
            end
        end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.CuArray(convert(Array{Complex{FloatGPU}, 2}, QZ))

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Temporary arrays:
    Ftmp = CuArrays.zeros(Complex{FloatGPU}, (grid.Nx, grid.Ny))

    # Problem:
    p = (responses, Ftmp, guard, QPARAXIAL, QZ, grid.FT)
    pfunc = Equations.PFunction(_func_xy!, p)
    prob = Equations.Problem(ALG, Ftmp, pfunc)

    return NonlinearPropagator(prob)
end


function propagate!(
    E::AbstractArray,
    NP::NonlinearPropagator,
    z::T,
    dz::T,
) where T<:AbstractFloat
    args = ()
    znew = NP.prob.step(E, z, dz, args)
    znext = z + dz
    while znew < znext
        dznew = znext - znew
        znew = NP.prob.step(E, znew, dznew, args)
    end
end


function _func_r!(
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


function _func_t!(
    dS::AbstractArray{Complex{T}, 1},
    S::AbstractArray{Complex{T}, 1},
    z::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    responses, FT, Etmp, Ftmp, Stmp, guard, QZ = p

    Fourier.hilbert!(Etmp, FT, S)   # spectrum real to signal analytic

    fill!(dS, 0)

    for resp in responses
        resp.calculate(Ftmp, Etmp, z, args)
        Guards.apply_field_filter!(guard, Ftmp)
        Fourier.rfft!(Stmp, FT, Ftmp)   # time -> frequency
        @. dS = dS + resp.Rnl * Stmp
    end

    @. dS = -1im * QZ * dS
    return nothing
end


function _func_rt!(
    dS::CuArrays.CuArray{Complex{T}, 2},
    S::CuArrays.CuArray{Complex{T}, 2},
    z::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    responses, FT, Etmp, Ftmp, Stmp, guard, QPARAXIAL, QZ, HT = p

    Fourier.hilbert!(Etmp, FT, S)   # spectrum real to signal analytic

    fill!(dS, 0)

    for resp in responses
        resp.calculate(Ftmp, Etmp, z, args)
        Guards.apply_field_filter!(guard, Ftmp)
        Fourier.rfft!(Stmp, FT, Ftmp)   # time -> frequency
        _update_dS!(dS, resp.Rnl, Stmp)   # dS = dS + Ra * Stmp
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


function _func_xy!(
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
        Fourier.fft!(dE, FT)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(guard, dE)
        Fourier.ifft!(dE, FT)
    end

    return nothing
end


function _update_dS!(
    dS::CuArrays.CuArray{Complex{T}, 2},
    R::T,
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    @. dS = dS + R * S
    return nothing
end


function _update_dS!(
    dS::CuArrays.CuArray{Complex{T}, 2},
    R::CuArrays.CuArray{Complex{T}, 1},
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    N1, N2 = size(S)
    nth = min(N1 * N2, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N1 * N2 / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _update_dS_kernel(dS, R, S)
    return nothing
end


function _update_dS_kernel(dS, R, S)
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
