# ******************************************************************************
# Lattice
# ******************************************************************************
function init_lattice(unit, grid, field, medium, plasma, args)
    refractive_index_perturbation = args["refractive_index_perturbation"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))

    n = @. Media.refractive_index((medium, ), grid.w * unit.w)
    mu = medium.permeability(grid.w * unit.w)
    Rnl = @. n / (MU0 * mu * C0^2) * Eu

    @. Rnl = conj(Rnl)
    Rnl = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Rnl))

    dn = refractive_index_perturbation(grid.r, unit.r)
    dn = CuArrays.CuArray(convert(Array{FloatGPU, 1}, dn))

    p = (dn, )

    return Rnl, calc_lattice, p
end


function calc_lattice(F::CuArrays.CuArray{FloatGPU, 2},
                      E::CuArrays.CuArray{ComplexGPU, 2},
                      p::Tuple)
    dn = p[1]
    @. F = (2 * dn + dn^2) * real(E)
    return nothing
end
