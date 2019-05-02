# ******************************************************************************
# Lattice
# ******************************************************************************
function init_lattice(unit, grid, field, medium, args)
    dnr_func = args["dnr_func"]
    dnz_func = args["dnz_func"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))

    n = @. Media.refractive_index((medium, ), grid.w * unit.w)
    mu = medium.permeability(grid.w * unit.w)
    Rnl = @. n / (MU0 * mu * C0^2) * Eu

    @. Rnl = conj(Rnl)
    Rnl = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Rnl))

    dnr = @. dnr_func(grid.r, unit.r)
    dnr = CuArrays.CuArray(convert(Array{FloatGPU, 1}, dnr))

    p = (dnr, dnz_func, unit.z)

    return Rnl, calc_lattice, p
end


function calc_lattice(z::T,
                      F::CuArrays.CuArray{T},
                      E::CuArrays.CuArray{Complex{T}},
                      p::Tuple) where T
    dnr = p[1]
    dnz_func = p[2]
    zu = p[3]

    dnz = dnz_func(z, zu)

    @. F = (2 * (dnr * dnz) + (dnr * dnz)^2) * real(E)
    return nothing
end
