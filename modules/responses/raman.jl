# ******************************************************************************
# Stimulated Raman nonlinearity
# ******************************************************************************
function Raman(unit, grid, field, medium, keys, guard)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    chi3 = Media.chi3_func(medium, field.w0)
    Rk = EPS0 * chi3 * Eu^3

    graman = medium.graman
    Rr = graman * Rk
    Rk = (1. - graman) * Rk   # Error!!!

    Rnl = CuArrays.cuzeros(ComplexGPU, grid.Nw)
    fill!(Rnl, FloatGPU(Rr))

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put
    # H(t) into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. medium.raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * grid.dt * unit.t

    if abs(1. - sum(Hraman)) > 1e-3
        println("WARNING: The integral of Raman response function should be" *
                " normalized to 1.")
    end

    Tguard = convert(Array{ComplexF64, 1}, CuArrays.collect(guard.T))
    @. Hraman = Hraman * Tguard   # temporal filter
    Hraman = Fourier.ifftshift(Hraman)
    Hramanw = FFTW.rfft(Hraman)   # time -> frequency

    Hramanw = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Hramanw))

    p = (keys["RTHG"], Hramanw, grid.FT)

    return NonlinearResponse(Rnl, func_raman, p)
end


function func_raman(F::CuArrays.CuArray{FloatGPU, 2},
                    E::CuArrays.CuArray{ComplexGPU, 2},
                    p::Tuple)
    THG = p[1]
    Hramanw = p[2]
    FT = p[3]
    if THG != 0
        @. F = real(E)^2
    else
        @. F = FloatGPU(3. / 4.) * abs2(E)
    end
    Fourier.convolution2!(FT, Hramanw, F)
    @. F = F * real(E)
    return nothing
end
