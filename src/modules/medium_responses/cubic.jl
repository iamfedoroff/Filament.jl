# ******************************************************************************
# Cubic nonlinear response
# ******************************************************************************
function init_cubic(unit, grid, field, medium, p)
    THG = p["THG"]
    n2 = p["n2"]

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4. / 3. * real(n0)^2 * EPS0 * C0 * n2
    Rnl = EPS0 * chi3 * Eu^3
    Rnl = FloatGPU(Rnl)

    if THG
        calc = calc_cubic
    else
        calc = calc_cubic_nothg
    end

    p_calc = ()

    mu = medium.permeability(w0)
    k0 = Media.k_func(medium, w0)
    QZ0 = MU0 * mu * w0^2 / (2. * k0) * unit.z / Eu
    Rnl0 = EPS0 * chi3 * 3. / 4. * Eu^3
    phi = QZ0 * abs(Rnl0)

    p_dzadapt = (phi, field.E)

    return Rnl, calc, p_calc, dzadapt_cubic, p_dzadapt
end


function calc_cubic(z::T,
                    F::CuArrays.CuArray{T, 2},
                    E::CuArrays.CuArray{Complex{T}, 2},
                    p::Tuple) where T
    @. F = real(E)^3
    return nothing
end


function calc_cubic_nothg(z::T,
                          F::CuArrays.CuArray{T, 2},
                          E::CuArrays.CuArray{Complex{T}, 2},
                          p::Tuple) where T
    @. F = 3 / 4 * abs2(E) * real(E)
    return nothing
end


function calc_cubic_nothg(z::T,
                          F::CuArrays.CuArray{Complex{T}},
                          E::CuArrays.CuArray{Complex{T}},
                          p::Tuple) where T
    @. F = 3 / 4 * abs2(E) * E
    return nothing
end


function dzadapt_cubic(phimax::AbstractFloat, p::Tuple)
    phi = p[1]
    E = p[2]
    Imax = maximum(abs2.(E))
    return phimax / (phi * Imax)
end