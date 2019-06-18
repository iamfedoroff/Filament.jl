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

    mu = medium.permeability(w0)
    k0 = Media.k_func(medium, w0)
    QZ0 = MU0 * mu * w0^2 / (2. * k0) * unit.z / Eu
    Rnl0 = 1im / w0 * QE^2 / MR / (nuc - 1im * w0) * unit.rho * Eu
    phi = QZ0 * abs(real(Rnl0))

    p_dzadapt = (phi, field.rho)

    return Rnl, calc_current_free, p_calc, dzadapt_current_free, p_dzadapt
end


function calc_current_free(z::T,
                           F::CuArrays.CuArray{T},
                           E::CuArrays.CuArray{Complex{T}},
                           p::Tuple) where T
    rho, = p
    @. F = rho * real(E)
    return nothing
end


function dzadapt_current_free(phimax::AbstractFloat, p::Tuple)
    phi, rho = p
    rhomax = maximum(rho)
    return phimax / (phi * rhomax)
end
