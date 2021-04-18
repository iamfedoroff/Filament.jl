struct NonlinearPropagator{P<:ODEIntegrators.Integrator}
    integ :: P
end


function NonlinearPropagator(
    unit::Units.Unit,
    grid::Grids.Grid,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    responses_list,
    PARAXIAL,
    ALG,
)
    # Prefactor:
    QZ = QZfunc(unit, grid, medium, field, PARAXIAL)

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
    p = (responses, QZ, Ftmp, guard, field.PS, field.PT, PARAXIAL)
    prob = ODEIntegrators.Problem(func!, Ftmp, p)
    integ = ODEIntegrators.Integrator(prob, ALG)

    return NonlinearPropagator(integ)
end


function func!(
    dE::AbstractArray{Complex{T}},
    E::AbstractArray{Complex{T}},
    p::Tuple,
    z::T,
) where T<:AbstractFloat
    responses, QZ, Ftmp, guard, PS, PT, PARAXIAL = p

    inverse_transform_time!(E, PT)

    @. dE = 0
    for resp in responses
        resp.calculate(Ftmp, E, resp.p, z)
        Guards.apply_field_filter!(Ftmp, guard)
        real_signal_to_analytic_spectrum!(Ftmp, PT)
        update_dE!(dE, resp.Rnl, Ftmp)   # dE = dE + Rnl * Ftmp
    end

    if PARAXIAL
        @. dE = 1im * QZ * dE
    else
        forward_transform_space!(dE, PS)
        @. dE = 1im * QZ * dE
        Guards.apply_spectral_filter!(dE, guard)
        inverse_transform_space!(dE, PS)
    end

    forward_transform_time!(E, PT)
    return nothing
end


function propagate!(
    E::AbstractArray, NP::NonlinearPropagator, z::T, dz::T,
) where T<:AbstractFloat
    ODEIntegrators.step!(NP.integ, E, z, dz)
end


# ******************************************************************************
function QZfunc(
    unit::Units.UnitR,
    grid::Grids.GridR,
    medium::Media.Medium,
    field::Fields.Field,
    PARAXIAL,
)
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, grid.Nr)
    for i=1:grid.Nr
        kt = grid.k[i] * unit.k
        QZ[i] = Qfunc(PARAXIAL, medium, w0, kt) * unit.z / Eu
    end

    return CUDA.CuArray{Complex{FloatGPU}}(QZ)
end


function QZfunc(
    unit::Units.UnitR,
    grid::Grids.GridRn,
    medium::Media.Medium,
    field::Fields.Field,
    PARAXIAL,
)
    @assert PARAXIAL

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = Qfunc(PARAXIAL, medium, w0, 0) * unit.z / Eu

    return QZ
end


function QZfunc(
    unit::Units.UnitT,
    grid::Grids.GridT,
    medium::Media.Medium,
    field::Fields.Field,
    PARAXIAL,
)
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, grid.Nt)
    for i=1:grid.Nt
        w = grid.w[i] * unit.w
        QZ[i] = Qfunc(PARAXIAL, medium, w, 0.0) * unit.z / Eu
    end

    return QZ
end


function QZfunc(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    medium::Media.Medium,
    field::Fields.Field,
    PARAXIAL,
)
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, (grid.Nr, grid.Nt))
    for j=1:grid.Nt
    for i=1:grid.Nr
        kt = grid.k[i] * unit.k
        w = grid.w[j] * unit.w
        QZ[i, j] = Qfunc(PARAXIAL, medium, w, kt) * unit.z / Eu
    end
    end
    return CUDA.CuArray{Complex{FloatGPU}}(QZ)
end


function QZfunc(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    medium::Media.Medium,
    field::Fields.Field,
    PARAXIAL,
)
    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    for j=1:grid.Ny
    for i=1:grid.Nx
        kt = sqrt((grid.kx[i] * unit.kx)^2 + (grid.ky[j] * unit.ky)^2)
        QZ[i, j] = Qfunc(PARAXIAL, medium, w0, kt) * unit.z / Eu
    end
    end
    return CUDA.CuArray{Complex{FloatGPU}}(QZ)
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
    dE::CUDA.CuArray{Complex{T}, 2},
    R::CUDA.CuArray{Complex{T}, 1},
    F::CUDA.CuArray{Complex{T}, 2},
) where T
    N = length(F)

    ckernel = CUDA.@cuda launch=false update_dE_kernel(dE, R, F)
    config = CUDA.launch_configuration(ckernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    ckernel(dE, R, F; threads=threads, blocks=blocks)
    return nothing
end


function update_dE_kernel(dE, R, F)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nr, Nt = size(F)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        dE[i, j] = dE[i, j] + R[j] * F[i, j]
    end
    return nothing
end
