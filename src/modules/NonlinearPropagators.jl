module NonlinearPropagators

import CuArrays
import CUDAnative
import HankelTransforms

import AnalyticSignals
import Constants: FloatGPU, MAX_THREADS_PER_BLOCK, MU0
import Equations
import Fields
import FourierTransforms
import Grids
import Guards
import Media
import Units


struct NonlinearPropagator
    integ :: Equations.Integrator
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
    Eu = Units.E(unit, real(n0))
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
    QZ = CuArrays.CuArray{Complex{FloatGPU}}(QZ)

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
    p = (responses, Ftmp, guard, QPARAXIAL, QZ, field.HT)
    prob = Equations.Problem(_func_r!, Ftmp, p)
    integ = Equations.Integrator(prob, ALG)

    return NonlinearPropagator(integ)
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
    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. 0.5 * MU0 * mu * (grid.w * unit.w)^2 * unit.z / Eu

    QZ = zeros(ComplexF64, grid.Nt)
    for iw=1:grid.Nt
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

    Ftmp = zeros(ComplexF64, grid.Nt)   # temporary array

    # Problem:
    p = (responses, field.FT, Ftmp, guard, QZ)
    prob = Equations.Problem(_func_t!, Ftmp, p)
    integ = Equations.Integrator(prob, ALG)

    return NonlinearPropagator(integ)
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
    Eu = Units.E(unit, real(n0))
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. 0.5 * MU0 * mu * (grid.w * unit.w)^2 * unit.z / Eu

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nt))
    if QPARAXIAL
        for iw=1:grid.Nt
            if beta[iw] != 0
                for ir=1:grid.Nr
                    QZ[ir, iw] = Qfactor[iw] / beta[iw]
                end
            end
        end
    else
        for iw=1:grid.Nt
        for ir=1:grid.Nr
            kzij = sqrt(beta[iw]^2 - (grid.k[ir] * unit.k)^2 + 0im)
            if kzij != 0
                QZ[ir, iw] = Qfactor[iw] / kzij
            end
        end
        end
    end
    @. QZ = conj(QZ)
    QZ = CuArrays.CuArray{Complex{FloatGPU}}(QZ)

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    Ftmp = CuArrays.zeros(Complex{FloatGPU}, (grid.Nr, grid.Nt))   # temporary array

    # Problem:
    p = (responses, field.FT, Ftmp, guard, QPARAXIAL, QZ, field.HT)
    prob = Equations.Problem(_func_rt!, Ftmp, p)
    integ = Equations.Integrator(prob, ALG)

    return NonlinearPropagator(integ)
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
    Eu = Units.E(unit, real(n0))
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
    QZ = CuArrays.CuArray{Complex{FloatGPU}}(QZ)

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
    p = (responses, Ftmp, guard, QPARAXIAL, QZ, field.FT)
    prob = Equations.Problem(_func_xy!, Ftmp, p)
    integ = Equations.Integrator(prob, ALG)

    return NonlinearPropagator(integ)
end


function propagate!(
    E::AbstractArray, NP::NonlinearPropagator, z::T, dz::T,
) where T<:AbstractFloat
    Equations.step!(NP.integ, E, z, dz)
end


function _func_r!(
    dE::AbstractArray{Complex{T}, 1},
    E::AbstractArray{Complex{T}, 1},
    p::Tuple,
    z::T,
) where T<:AbstractFloat
    responses, Ftmp, guard, QPARAXIAL, QZ, HT = p

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL
        @. dE = -1im * QZ * dE
    else
        HankelTransforms.dht!(dE, HT)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(dE, guard)
        HankelTransforms.idht!(dE, HT)
    end

    return nothing
end


function _func_t!(
    dE::AbstractArray{Complex{T}, 1},
    E::AbstractArray{Complex{T}, 1},
    p::Tuple,
    z::T,
) where T<:AbstractFloat
    responses, FT, Ftmp, guard, QZ = p

    FourierTransforms.ifft!(E, FT)

    fill!(dE, 0)
    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        AnalyticSignals.rsig2aspec!(Ftmp, FT)
        @. dE = dE + resp.Rnl * Ftmp
    end
    @. dE = -1im * QZ * dE

    FourierTransforms.fft!(E, FT)
    return nothing
end


function _func_rt!(
    dE::AbstractArray{Complex{T}, 2},
    E::AbstractArray{Complex{T}, 2},
    p::Tuple,
    z::T,
) where T<:AbstractFloat
    responses, FT, Ftmp, guard, QPARAXIAL, QZ, HT = p

    FourierTransforms.ifft!(E, FT)

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        AnalyticSignals.rsig2aspec!(Ftmp, FT)
        _update_dE!(dE, resp.Rnl, Ftmp)   # dE = dE + Ra * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL
        @. dE = -1im * QZ * dE
    else
        HankelTransforms.dht!(dE, HT)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(dE, guard)
        HankelTransforms.idht!(dE, HT)
    end

    FourierTransforms.fft!(E, FT)
    return nothing
end


function _func_xy!(
    dE::AbstractArray{Complex{T}, 2},
    E::AbstractArray{Complex{T}, 2},
    p::Tuple,
    z::T,
) where T<:AbstractFloat
    responses, Ftmp, guard, QPARAXIAL, QZ, FT = p

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if QPARAXIAL
        @. dE = -1im * QZ * dE
    else
        FourierTransforms.fft!(dE, FT)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(dE, guard)
        FourierTransforms.ifft!(dE, FT)
    end

    return nothing
end


function _update_dE!(
    dE::CuArrays.CuArray{Complex{T}, 2},
    R::T,
    E::CuArrays.CuArray{Complex{T}, 2},
) where T
    @. dE = dE + R * E
    return nothing
end


function _update_dE!(
    dE::CuArrays.CuArray{Complex{T}, 2},
    R::CuArrays.CuArray{Complex{T}, 1},
    E::CuArrays.CuArray{Complex{T}, 2},
) where T
    N1, N2 = size(E)
    nth = min(N1 * N2, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N1 * N2 / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _update_dE_kernel(dE, R, E)
    return nothing
end


function _update_dE_kernel(dE, R, E)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(E)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        dE[i, j] = dE[i, j] + R[j] * E[i, j]
    end
    return nothing
end


end
