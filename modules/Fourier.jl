module Fourier

struct FourierTransform
    Nt :: Int64
    Nw :: Int64
    PFFT :: Base.DFT.Plan
    PIFFT :: Base.DFT.Plan
    PRFFT :: Base.DFT.Plan
    PIRFFT :: Base.DFT.Plan
end

function FourierTransform(Nt, FFTWFLAG)
    if iseven(Nt)   # Nt is even
        Nw = div(Nt, 2) + 1
    else   # Nt is odd
        Nw = div(Nt + 1, 2)
    end
    PFFT = plan_fft(zeros(Complex128, Nt); flags=FFTWFLAG)
    PIFFT = plan_ifft(zeros(Complex128, Nt); flags=FFTWFLAG)
    PRFFT = plan_rfft(zeros(Float64, Nt); flags=FFTWFLAG)
    PIRFFT = plan_irfft(zeros(Complex128, Nw), Nt; flags=FFTWFLAG)
    return FourierTransform(Nt, Nw, PFFT, PIFFT, PRFFT, PIRFFT)
end


function fft1d(FT::FourierTransform, E::Array{Complex128, 1})
    S = zeros(Complex128, FT.Nt)
    A_mul_B!(S, FT.PIFFT, E)   # time -> frequency
    @inbounds @. S = S * FT.Nt
    return S
end


function ifft1d(FT::FourierTransform, S::Array{Complex128, 1})
    E = zeros(Complex128, FT.Nt)
    A_mul_B!(E, FT.PFFT, S)   # frequency -> time
    @inbounds @. E = E / FT.Nt
    return E
end


function rfft1d(FT::FourierTransform, E::Array{Float64, 1})
    S = zeros(Complex128, FT.Nw)
    A_mul_B!(S, FT.PRFFT, E)   # time -> frequency
    @inbounds @. S = conj(S)
    return S
end


function irfft1d(FT::FourierTransform, S::Array{Complex128, 1})
    E = zeros(Float64, FT.Nt)
    Sconj = zeros(Complex128, FT.Nw)
    @inbounds @. Sconj = conj(S)
    A_mul_B!(E, FT.PIRFFT, Sconj)   # frequency -> time
    return E
end


"""Real time signal -> analytic time signal."""
function signal_real_to_analytic(Er::Array{Float64, 1})
    N = length(Er)
    S = ifft(Er) * N   # time -> frequency
    if iseven(N)   # N is even
        S[2:div(N, 2)] = 2. * S[2:div(N, 2)]
        S[div(N, 2) + 2:end] = 0.
    else   # N is odd
        S[2:div(N + 1, 2)] = 2. * S[2:div(N + 1, 2)]
        S[div(N + 1, 2) + 1:end] = 0.
    end
    Ea = fft(S) / N   # frequency -> time
    return Ea
end


"""Spectrum of real time signal -> spectrum of analytic time signal."""
function spectrum_real_to_analytic(S::Array{Complex128, 1}, Nt::Int64)
    Sa = zeros(Complex128, Nt)
    Sa[1] = S[1]
    if iseven(Nt)   # Nt is even
        @inbounds @views @. Sa[2:div(Nt, 2)] = 2. * S[2:div(Nt, 2)]
        Sa[div(Nt, 2) + 1] = S[div(Nt, 2) + 1]
    else   # Nt is odd
        @inbounds @views @. Sa[2:div(Nt + 1, 2)] = 2. * S[2:div(Nt + 1, 2)]
    end
    return Sa
end


"""
Roll array elements. Elements that roll beyond the last position are
re-introduced at the first. With nroll = (Nt + 1) / 2 the function is equivalent
to fftshift function from NumPy library for the Python.
"""
function roll(a::Array{Float64, 1}, nroll::Int64)
    N = length(a)
    aroll = zeros(N)
    for i=1:N
        ii = mod(i + nroll, N)
        if ii == 0
            ii = N
        end
        aroll[ii] = a[i]
    end
    return aroll
end


end
