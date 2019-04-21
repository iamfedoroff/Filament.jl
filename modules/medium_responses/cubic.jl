# ******************************************************************************
# Cubic nonlinear response
# ******************************************************************************
function init_cubic(unit, grid, field, medium, plasma, args)
    THG = args["THG"]
    n2 = args["n2"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4. / 3. * real(n0)^2 * EPS0 * C0 * n2
    Rnl = EPS0 * chi3 * Eu^3
    Rnl = FloatGPU(Rnl)

    p = ()

    if THG
        calc = calc_cubic
    else
        calc = calc_cubic_nothg
    end

    return Rnl, calc, p
end


function calc_cubic(z::T,
                    F::CuArrays.CuArray{T},
                    E::CuArrays.CuArray{Complex{T}},
                    p::Tuple) where T
    @. F = real(E)^3
    return nothing
end


function calc_cubic_nothg(z::T,
                          F::CuArrays.CuArray{T},
                          E::CuArrays.CuArray{Complex{T}},
                          p::Tuple) where T
    @. F = 3 / 4 * abs2(E) * real(E)
    return nothing
end
