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

    if THG
        calc = calc_cubic
    else
        calc = calc_cubic_nothg
    end

    p_calc = ()
    pcalc = Equations.PFunction(calc, p_calc)
    return Media.NonlinearResponse(Rnl, pcalc)
end


function calc_cubic(F::AbstractArray{T, 2},
                    E::AbstractArray{Complex{T}, 2},
                    z::T,
                    args::Tuple,
                    p::Tuple) where T<:AbstractFloat
    @. F = real(E)^3
    return nothing
end


function calc_cubic_nothg(F::AbstractArray{T, 2},
                          E::AbstractArray{Complex{T}, 2},
                          z::T,
                          args::Tuple,
                          p::Tuple) where T<:AbstractFloat
    @. F = 3 / 4 * abs2(E) * real(E)
    return nothing
end


function calc_cubic_nothg(F::AbstractArray{Complex{T}},
                          E::AbstractArray{Complex{T}},
                          z::T,
                          args::Tuple,
                          p::Tuple) where T<:AbstractFloat
    @. F = 3 / 4 * abs2(E) * E
    return nothing
end
