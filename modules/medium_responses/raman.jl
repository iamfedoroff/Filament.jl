# ******************************************************************************
# Stimulated Raman response
# ******************************************************************************
function init_raman(unit, grid, field, medium, plasma, args)
    RTHG = args["RTHG"]
    n2 = args["n2"]
    raman_response = args["raman_response"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4. / 3. * real(n0)^2 * EPS0 * C0 * n2
    Rr = EPS0 * chi3 * Eu^3

    Rnl = CuArrays.cuzeros(ComplexGPU, grid.Nw)
    fill!(Rnl, FloatGPU(Rr))

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put
    # H(t) into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * grid.dt * unit.t

    if abs(1. - sum(Hraman)) > 1e-3
        println("WARNING: The integral of Raman response function should be" *
                " normalized to 1.")
    end

    # Tguard = convert(Array{ComplexF64, 1}, CuArrays.collect(guard.T))
    # @. Hraman = Hraman * Tguard   # temporal filter
    Hraman = Fourier.ifftshift(Hraman)
    Hramanw = FFTW.rfft(Hraman)   # time -> frequency

    Hramanw = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Hramanw))

    p = (RTHG, Hramanw, grid.FT)

    return NonlinearResponses.NonlinearResponse(Rnl, calculate_raman, p)
end


function calculate_raman(F::CuArrays.CuArray{FloatGPU, 2},
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
