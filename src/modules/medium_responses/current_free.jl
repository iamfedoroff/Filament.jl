# ******************************************************************************
# Plasma nonlinearity
# ******************************************************************************
function init_current_free(unit, grid, field, medium, p)
    nuc = p["nuc"]
    mr = p["mr"]

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))
    MR = mr * ME   # reduced mass of electron and hole (effective mass)

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

    p_calc = (field.rho, )
    pcalc = Equations.PFunction(calc_current_free, p_calc)
    return Media.NonlinearResponse(Rnl, pcalc)
end


function calc_current_free(F::AbstractArray{T},
                           E::AbstractArray{Complex{T}},
                           args::Tuple,
                           p::Tuple) where T<:AbstractFloat
    rho, = p
    @. F = rho * real(E)
    return nothing
end
