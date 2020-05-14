struct NonlinearPropagator{P<:Equations.Integrator}
    integ :: P
end


# ******************************************************************************
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
    p = (responses, QZ, Ftmp, guard, nothing, field.HT, PARAXIAL)
    prob = Equations.Problem(func!, Ftmp, p)
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
    p = (responses, QZ, Ftmp, guard, field.FT, nothing, PARAXIAL)
    prob = Equations.Problem(func!, Ftmp, p)
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
    p = (responses, QZ, Ftmp, guard, field.FT, field.HT, PARAXIAL)
    prob = Equations.Problem(func!, Ftmp, p)
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
    p = (responses, QZ, Ftmp, guard, nothing, field.FT, PARAXIAL)
    prob = Equations.Problem(func!, Ftmp, p)
    integ = Equations.Integrator(prob, ALG)

    return NonlinearPropagator(integ)
end


function propagate!(
    E::AbstractArray, NP::NonlinearPropagator, z::T, dz::T,
) where T<:AbstractFloat
    Equations.step!(NP.integ, E, z, dz)
end


function func!(
    dE::AbstractArray{Complex{T}},
    E::AbstractArray{Complex{T}},
    p::Tuple,
    z::T,
) where T<:AbstractFloat
    responses, QZ, Ftmp, guard, PT, PS, PARAXIAL = p

    inverse_transform_time!(E, PT)

    @. dE = 0
    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        real_signal_to_analytic_spectrum!(Ftmp, PT)
        update_dE!(dE, resp.Rnl, Ftmp)   # dE = dE + Rnl * Ftmp
    end

    if PARAXIAL
        @. dE = -1im * QZ * dE
    else
        forward_transform_space!(dE, PS)
        @. dE = -1im * QZ * dE
        Guards.apply_spectral_filter!(dE, guard)
        inverse_transform_space!(dE, PS)
    end

    forward_transform_time!(E, PT)
    return nothing
end


# ******************************************************************************
function update_dE!(
    dE::AbstractArray{Complex{T}},
    R::Union{T, AbstractArray{T}, AbstractArray{Complex{T}}},
    F::AbstractArray{Complex{T}},
) where T
    @. dE = dE + R * F
    return nothing
end


function update_dE!(
    dE::CuArrays.CuArray{Complex{T}, 2},
    R::CuArrays.CuArray{Complex{T}, 1},
    F::CuArrays.CuArray{Complex{T}, 2},
) where T
    N = length(F)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDAdrv.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDAnative.@cuda config=get_config update_dE_kernel(dE, R, F)
    return nothing
end


function update_dE_kernel(dE, R, F)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(F)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        dE[i, j] = dE[i, j] + R[j] * F[i, j]
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