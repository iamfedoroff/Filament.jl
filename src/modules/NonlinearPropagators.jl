module NonlinearPropagators

import CuArrays
import CUDAnative
import HankelTransforms

import ..AnalyticSignals
import ..FourierTransforms
import ..Equations

import ..Constants: FloatGPU, MAX_THREADS_PER_BLOCK, MU0
import ..Fields
import ..Grids
import ..Guards
import ..Media
import ..Units


struct NonlinearPropagator{P<:Equations.Integrator}
    integ :: P
end


function NonlinearPropagator(
    unit::Units.UnitR,
    grid::Grids.GridR,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    responses_list,
    PARAXIAL,
    ALG,
)
    # Prefactor:
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, grid.Nr)
    for i=1:grid.Nr
        kt = grid.k[i] * unit.k
        QZ[i] = Qfunc(PARAXIAL, medium, w0, kt) * unit.z / Eu
        QZ[i] = conj(QZ[i])   # in order to make fft instead of ifft
    end
    QZ = CuArrays.CuArray{Complex{FloatGPU}}(QZ)

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Problem:
    Ftmp = zero(field.E)
    p = (responses, Ftmp, guard, PARAXIAL, QZ, field.HT)
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
    PARAXIAL,
    ALG,
)
    # Prefactor:
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, grid.Nt)
    for i=1:grid.Nt
        w = grid.w[i] * unit.w
        QZ[i] = Qfunc(PARAXIAL, medium, w, 0.0) * unit.z / Eu
        QZ[i] = conj(QZ[i])   # in order to make fft instead of ifft
    end

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Problem:
    Ftmp = zero(field.E)
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
    PARAXIAL,
    ALG,
)
    # Prefactor:
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nt))
    for j=1:grid.Nt
    for i=1:grid.Nr
        kt = grid.k[i] * unit.k
        w = grid.w[j] * unit.w
        QZ[i, j] = Qfunc(PARAXIAL, medium, w, kt) * unit.z / Eu
        QZ[i, j] = conj(QZ[i, j])   # in order to make fft instead of ifft
    end
    end
    QZ = CuArrays.CuArray{Complex{FloatGPU}}(QZ)

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Problem:
    Ftmp = zero(field.E)
    p = (responses, field.FT, Ftmp, guard, PARAXIAL, QZ, field.HT)
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
    PARAXIAL,
    ALG,
)
    # Prefactor:
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    for j=1:grid.Ny
    for i=1:grid.Nx
        kt = sqrt((grid.kx[i] * unit.kx)^2 + (grid.ky[j] * unit.ky)^2)
        QZ[i, j] = Qfunc(PARAXIAL, medium, w0, kt) * unit.z / Eu
        QZ[i, j] = conj(QZ[i, j])   # in order to make fft instead of ifft
    end
    end
    QZ = CuArrays.CuArray{Complex{FloatGPU}}(QZ)

    # Responses:
    responses = []
    for item in responses_list
        init = item["init"]
        response = init(unit, grid, field, medium, item)
        push!(responses, response)
    end
    responses = tuple(responses...)

    # Problem:
    Ftmp = zero(field.E)
    p = (responses, Ftmp, guard, PARAXIAL, QZ, field.FT)
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
    responses, Ftmp, guard, PARAXIAL, QZ, HT = p

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if PARAXIAL
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
    responses, FT, Ftmp, guard, PARAXIAL, QZ, HT = p

    FourierTransforms.ifft!(E, FT)

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        AnalyticSignals.rsig2aspec!(Ftmp, FT)
        _update_dE!(dE, resp.Rnl, Ftmp)   # dE = dE + Ra * Ftmp
    end

    # Nonparaxiality:
    if PARAXIAL
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
    responses, Ftmp, guard, PARAXIAL, QZ, FT = p

    fill!(dE, 0)

    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        @. dE = dE + resp.Rnl * Ftmp
    end

    # Nonparaxiality:
    if PARAXIAL
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
    nbl = cld(N1 * N2, nth)
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


# ******************************************************************************
function Qfunc(PARAXIAL, medium, w, kt)
    if PARAXIAL
        Q = Qfunc_paraxial(medium, w, kt)
    else
        Q = Qfunc_nonparaxial(medium, w, kt)
    end
    return Q
end


function Qfunc_paraxial(medium, w, kt)
    mu = medium.permeability(w)
    beta = Media.beta_func(medium, w)
    if beta == 0
        Q = zero(kt)
    else
        Q = MU0 * mu * w^2 / (2 * beta)
    end
    return Q
end


function Qfunc_nonparaxial(medium, w, kt)
    mu = medium.permeability(w)
    beta = Media.beta_func(medium, w)
    kz = sqrt(beta^2 - kt^2 + 0im)
    if kz == 0
        Q = zero(kt)
    else
        Q = MU0 * mu * w^2 / (2 * kz)
    end
    return Q
end


end
