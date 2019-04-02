# ******************************************************************************
# Stimulated Raman response
# ******************************************************************************
function init_raman(unit, grid, field, medium, plasma, args)
    THG = args["THG"]
    n2 = args["n2"]
    raman_response = args["raman_response"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4. / 3. * real(n0)^2 * EPS0 * C0 * n2
    Rnl = EPS0 * chi3 * Eu^3
    Rnl = FloatGPU(Rnl)

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

    p = (Hramanw, grid.FT)

    if THG
        calc = calc_raman
    else
        calc = calc_raman_nothg
    end

    return Rnl, calc, p
end


function calc_raman(z::Float64,
                    F::CuArrays.CuArray{FloatGPU, 2},
                    E::CuArrays.CuArray{ComplexGPU, 2},
                    p::Tuple)
    Hramanw, FT = p
    @. F = real(E)^2
    Fourier.convolution2!(FT, Hramanw, F)
    @. F = F * real(E)
    return nothing
end


function calc_raman_nothg(z::Float64,
                          F::CuArrays.CuArray{FloatGPU, 2},
                          E::CuArrays.CuArray{ComplexGPU, 2},
                          p::Tuple)
    Hramanw, FT = p
    @. F = FloatGPU(3. / 4.) * abs2(E)
    Fourier.convolution2!(FT, Hramanw, F)
    @. F = F * real(E)
    return nothing
end
