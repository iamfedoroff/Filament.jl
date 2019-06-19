# ******************************************************************************
# Stimulated Raman response
# ******************************************************************************
function init_raman(unit, grid, field, medium, p)
    THG = p["THG"]
    n2 = p["n2"]
    raman_response = p["raman_response"]

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

    Hraman = Fourier.ifftshift(Hraman)
    Hramanw = FFTW.rfft(Hraman)   # time -> frequency

    Hramanw = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Hramanw))

    if THG
        calc = calc_raman
    else
        calc = calc_raman_nothg
    end

    p_calc = (Hramanw, grid.FT)
    pcalc = PFunctions.PFunction(calc, p_calc)

    p_dzadapt = ()
    pdzadapt = PFunctions.PFunction(dzadapt_raman, p_dzadapt)

    return Media.NonlinearResponse(Rnl, pcalc, pdzadapt)
end


function calc_raman(F::CuArrays.CuArray{T},
                    E::CuArrays.CuArray{Complex{T}},
                    args::Tuple,
                    p::Tuple) where T
    Hramanw, FT = p
    @. F = real(E)^2
    Fourier.convolution2!(FT, Hramanw, F)
    @. F = F * real(E)
    return nothing
end


function calc_raman_nothg(F::CuArrays.CuArray{T},
                          E::CuArrays.CuArray{Complex{T}},
                          args::Tuple,
                          p::Tuple) where T
    Hramanw, FT = p
    @. F = FloatGPU(3. / 4.) * abs2(E)
    Fourier.convolution2!(FT, Hramanw, F)
    @. F = F * real(E)
    return nothing
end


function dzadapt_raman(phimax::AbstractFloat, p::Tuple)
    return Inf
end
