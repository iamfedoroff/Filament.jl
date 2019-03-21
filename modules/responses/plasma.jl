# ******************************************************************************
# Plasma nonlinearity
# ******************************************************************************
function Plasma(unit, grid, field, medium, plasma)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    nuc = plasma.nuc
    MR = plasma.mr * ME   # reduced mass of electron and hole (effective mass)

    Rnl = zeros(ComplexF64, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            Rnl[i] = 1im / (grid.w[i] * unit.w) *
                     QE^2 / MR / (nuc - 1im * (grid.w[i] * unit.w)) *
                     unit.rho * Eu
        end
    end

    @. Rnl = conj(Rnl)
    Rnl = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Rnl))

    p = (plasma.rho, )

    return NonlinearResponse(Rnl, func_plasma, p)
end


function func_plasma(F::CuArrays.CuArray{FloatGPU, 2},
                     E::CuArrays.CuArray{ComplexGPU, 2},
                     p::Tuple)
    rho = p[1]
    @. F = rho * real(E)
    return nothing
end
