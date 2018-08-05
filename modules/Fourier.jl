module Fourier

using PyCall
@pyimport numpy.fft as npfft


struct FourierTransform
    Nt :: Int64
    Nw :: Int64
    HS :: Array{Float64, 1}
    Sc :: Array{Complex128, 1}
    Sr :: Array{Complex128, 1}
    PFFT :: Base.DFT.Plan
    PIFFT :: Base.DFT.Plan
    PRFFT :: Base.DFT.Plan
    PIRFFT :: Base.DFT.Plan
end


function FourierTransform(Nt, dt)
    if iseven(Nt)   # Nt is even
        Nw = div(Nt, 2) + 1
    else   # Nt is odd
        Nw = div(Nt + 1, 2)
    end

    f = npfft.fftfreq(Nt, dt)
    HS = 1. + sign.(f)   # Heaviside-like step function for Hilbert transform

    # arrays to store intermediate results:
    Sc = zeros(Complex128, Nt)
    Sr = zeros(Complex128, Nw)

    PFFT = plan_fft(zeros(Complex128, Nt))
    PIFFT = plan_ifft(zeros(Complex128, Nt))
    PRFFT = plan_rfft(zeros(Float64, Nt))
    PIRFFT = plan_irfft(zeros(Complex128, Nw), Nt)
    return FourierTransform(Nt, Nw, HS, Sc, Sr, PFFT, PIFFT, PRFFT, PIRFFT)
end


function fft1d!(FT::FourierTransform, Ec::Array{Complex128, 1},
                Sc::Array{Complex128, 1})
    A_mul_B!(Sc, FT.PFFT, Ec)
    return nothing
end


function fft1d(FT::FourierTransform, Ec::Array{Complex128, 1})
    Sc = zeros(Complex128, FT.Nt)
    fft1d!(FT, Ec, Sc)   # time -> frequency
    return Sc
end


function ifft1d!(FT::FourierTransform, Sc::Array{Complex128, 1},
                 Ec::Array{Complex128, 1})
    A_mul_B!(Ec, FT.PIFFT, Sc)
    return nothing
end


function ifft1d(FT::FourierTransform, Sc::Array{Complex128, 1})
    Ec = zeros(Complex128, FT.Nt)
    ifft1d!(FT, Sc, Ec)   # frequency -> time
    return Ec
end


function rfft1d!(FT::FourierTransform, Er::Array{Float64, 1},
                 Sr::Array{Complex128, 1})
    A_mul_B!(Sr, FT.PRFFT, Er)   # time -> frequency
    return nothing
end


function rfft1d(FT::FourierTransform, Er::Array{Float64, 1})
    Sr = zeros(Complex128, FT.Nw)
    rfft1d!(FT, Er, Sr)   # time -> frequency
    return Sr
end


function rfft2d!(FT::FourierTransform, E::Array{Complex128, 2},
                 S::Array{Complex128, 2})
    Nr, Nt = size(E)
    Et = zeros(Float64, FT.Nt)
    St = zeros(Complex128, FT.Nw)
    for i=1:Nr
        @inbounds @views @. Et = real(E[i, :])
        rfft1d!(FT, Et, St)   # time -> frequency
        @inbounds @. S[i, :] = St
    end
    return nothing
end


function irfft1d!(FT::FourierTransform, Sr::Array{Complex128, 1},
                  Er::Array{Float64, 1})
    A_mul_B!(Er, FT.PIRFFT, Sr)   # frequency -> time
    return Er
end


function irfft1d(FT::FourierTransform, Sr::Array{Complex128, 1})
    Er = zeros(Float64, FT.Nt)
    Sr2 = copy(Sr)
    irfft1d!(FT, Sr2, Er)   # frequency -> time
    return Er
end


"""Real time signal -> analytic time signal."""
function signal_real_to_signal_analytic(FT::FourierTransform,
                                        Er::Array{Float64, 1})
    # Need test for odd N and low frequencies
    S = fft(Er)
    @. S = FT.HS * S
    Ea = ifft(S)
    return Ea
end


"""Spectrum of real time signal -> analytic time signal."""
function spectrum_real_to_signal_analytic!(FT::FourierTransform,
                                           Sr::Array{Complex128, 1},
                                           Ea::Array{Complex128, 1})
    # Need test for odd N and low frequencies
    # S = vcat(Sr, conj(Sr[end-1:-1:2]))
    @inbounds @. FT.Sc[1:FT.Nw] = Sr
    @inbounds @. FT.Sc = FT.HS * FT.Sc
    ifft1d!(FT, FT.Sc, Ea)
    return nothing
end


function spectrum_real_to_signal_analytic_2d!(FT::FourierTransform,
                                              S::Array{Complex128, 2},
                                              E::Array{Complex128, 2})
    Nr, Nt = size(E)
    St = zeros(Complex128, FT.Nw)
    Et = zeros(Complex128, FT.Nt)
    for i=1:Nr
        @inbounds @views @. St = S[i, :]
        spectrum_real_to_signal_analytic!(FT, St, Et)
        @inbounds @views @. E[i, :] = Et
    end
end


function convolution!(FT::FourierTransform, Hw::Array{Complex128, 1},
                      x::Array{Float64, 1}, res::Array{Float64, 1})
    rfft1d!(FT, x, FT.Sr)
    @inbounds @. FT.Sr = Hw * FT.Sr
    irfft1d!(FT, FT.Sr, res)
    return nothing
end


end
