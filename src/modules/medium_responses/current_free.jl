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

    Rnl = zeros(ComplexF64, grid.Nt)
    for i=1:grid.Nt
        if grid.w[i] != 0
            Rnl[i] = 1im / (grid.w[i] * unit.w) *
                     QE^2 / MR / (nuc - 1im * (grid.w[i] * unit.w)) *
                     unit.rho * Eu
        end
    end

    if ! isa(grid, Grids.GridT)   # FIXME: should be removed in a generic code.
        Rnl = CuArrays.CuArray{Complex{FloatGPU}}(Rnl)
    end

    p = (field.rho, )
    return Media.NonlinearResponse(Rnl, calc_current_free, p)
end


function calc_current_free(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    rho, = p
    @. F = rho * real(E)
    return nothing
end
