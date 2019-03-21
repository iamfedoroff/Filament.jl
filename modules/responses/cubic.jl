# ******************************************************************************
# Kerr nonlinearity
# ******************************************************************************
function Kerr(unit, grid, field, medium, keys)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    chi3 = Media.chi3_func(medium, field.w0)
    Rk = EPS0 * chi3 * Eu^3

    graman = medium.graman
    if keys["RAMAN"] !=0
        Rk = (1. - graman) * Rk
    end

    Rnl = CuArrays.cuzeros(ComplexGPU, grid.Nw)
    fill!(Rnl, FloatGPU(Rk))

    p = (keys["THG"], )

    return NonlinearResponse(Rnl, func_kerr, p)
end


function func_kerr(F::CuArrays.CuArray{FloatGPU, 2},
                   E::CuArrays.CuArray{ComplexGPU, 2},
                   p::Tuple)
    THG = p[1]
    if THG != 0
        @. F = real(E)^3
    else
        @. F = FloatGPU(3. / 4.) * abs2(E) * real(E)
    end
    return nothing
end
