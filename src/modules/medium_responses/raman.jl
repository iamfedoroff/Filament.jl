# ******************************************************************************
# Stimulated Raman response
# ******************************************************************************
function init_raman(unit, grid, field, medium, p)
    THG = p["THG"]
    n2 = p["n2"]
    raman_response = p["raman_response"]

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))
    chi3 = 4 / 3 * real(n0)^2 * EPS0 * C0 * n2
    Rnl = EPS0 * chi3 * Eu^3
    Rnl = convert(FloatGPU, Rnl)

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put
    # H(t) into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * grid.dt * unit.t
    Hraman = FourierTransforms.ifftshift(Hraman)
    Hraman = @. Hraman + 0im   # real -> complex
    FFTW.fft!(Hraman)   # time -> frequency
    if !isa(grid, Grids.GridT)   # FIXME: should be removed in a generic code.
        Hraman = CuArrays.CuArray{Complex{FloatGPU}}(Hraman)
    end

    if THG
        calc = calc_raman
    else
        calc = calc_raman_nothg
    end

    p = (Hraman, field.FT)
    return Media.NonlinearResponse(Rnl, calc, p)
end


function calc_raman(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    Hraman, FT = p
    @. F = real(E)^2
    FourierTransforms.convolution!(F, FT, Hraman)
    @. F = F * real(E)
    return nothing
end


function calc_raman_nothg(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    Hraman, FT = p
    @. F = FloatGPU(3 / 4) * abs2(E)
    FourierTransforms.convolution!(F, FT, Hraman)
    @. F = F * real(E)
    return nothing
end
