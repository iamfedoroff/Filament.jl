# ******************************************************************************
# Lattice
# ******************************************************************************
function init_lattice(unit, grid, field, medium, p)
    dnr_func = p["dnr_func"]
    dnz_func = p["dnz_func"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))

    n = @. Media.refractive_index((medium, ), grid.w * unit.w)
    mu = medium.permeability(grid.w * unit.w)
    Rnl = @. n / (MU0 * mu * C0^2) * Eu

    @. Rnl = conj(Rnl)
    Rnl = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Rnl))

    dnr = @. dnr_func(grid.r, unit.r)
    dnr = CuArrays.CuArray(convert(Array{FloatGPU, 1}, dnr))

    p_calc = (dnr, dnz_func, unit.z)
    pcalc = Equations.PFunction(calc_lattice, p_calc)
    return Media.NonlinearResponse(Rnl, pcalc)
end


function calc_lattice(F::AbstractArray{T},
                      E::AbstractArray{Complex{T}},
                      z::T,
                      args::Tuple,
                      p::Tuple) where T<:AbstractFloat
    dnr, dnz_func, zu = p

    dnz = dnz_func(z, zu)

    @. F = (2 * (dnr * dnz) + (dnr * dnz)^2) * real(E)
    return nothing
end