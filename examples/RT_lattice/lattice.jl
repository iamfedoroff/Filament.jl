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

    Rnl = CuArrays.CuArray{Complex{FloatGPU}}(Rnl)

    dnr = @. dnr_func(grid.r, unit.r)
    dnr = CuArrays.CuArray{FloatGPU}(dnr)

    p = (dnr, dnz_func, unit.z)
    return Media.NonlinearResponse(Rnl, calc_lattice, p)
end


function calc_lattice(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    dnr, dnz_func, zu = p

    dnz = dnz_func(z, zu)

    @. F = (2 * (dnr * dnz) + (dnr * dnz)^2) * real(E)
    return nothing
end
