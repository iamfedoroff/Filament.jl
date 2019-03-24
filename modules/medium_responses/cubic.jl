# ******************************************************************************
# Cubic response
# ******************************************************************************
function init_cubic(unit, grid, field, medium, plasma, args)
    THG = args["THG"]
    n2 = args["n2"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4. / 3. * real(n0)^2 * EPS0 * C0 * n2
    Rk = EPS0 * chi3 * Eu^3

    Rnl = CuArrays.cuzeros(ComplexGPU, grid.Nw)
    fill!(Rnl, FloatGPU(Rk))

    p = (THG, )

    return NonlinearResponses.NonlinearResponse(Rnl, calculate_cubic, p)
end


function calculate_cubic(F::CuArrays.CuArray{FloatGPU, 2},
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
