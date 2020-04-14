# ******************************************************************************
# Cubic nonlinear response
# ******************************************************************************
function init_cubic(unit, grid, field, medium, p)
    THG = p["THG"]
    n2 = p["n2"]

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4 / 3 * real(n0)^2 * EPS0 * C0 * n2
    Rnl = EPS0 * chi3 * Eu^3
    Rnl = convert(FloatGPU, Rnl)

    if (typeof(grid) <: Grids.GridR) | (typeof(grid) <: Grids.GridXY)
        calc = calc_cubic_nothg_spatial
    else
        if THG
            calc = calc_cubic
        else
            calc = calc_cubic_nothg
        end
    end

    p = ()
    return Media.NonlinearResponse(Rnl, calc, p)
end


function calc_cubic(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    @. F = real(E)^3
    return nothing
end


function calc_cubic_nothg(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    @. F = 3 / 4 * abs2(E) * real(E)
    return nothing
end


function calc_cubic_nothg_spatial(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    @. F = 3 / 4 * abs2(E) * E
    return nothing
end
