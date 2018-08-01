module Fourier

struct FourierTransform
    Nt :: Int64
    Nw :: Int64
    PFFT :: Base.DFT.Plan
    PIFFT :: Base.DFT.Plan
    PRFFT :: Base.DFT.Plan
    PIRFFT :: Base.DFT.Plan
end

function FourierTransform(Nt)
    if iseven(Nt)   # Nt is even
        Nw = div(Nt, 2) + 1
    else   # Nt is odd
        Nw = div(Nt + 1, 2)
    end
    PFFT = plan_fft(zeros(Complex128, Nt))
    PIFFT = plan_ifft(zeros(Complex128, Nt))
    PRFFT = plan_rfft(zeros(Float64, Nt))
    PIRFFT = plan_irfft(zeros(Complex128, Nw), Nt)
    return FourierTransform(Nt, Nw, PFFT, PIFFT, PRFFT, PIRFFT)
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
function signal_real_to_analytic(Er::Array{Float64, 1})
    N = length(Er)
    S = fft(Er)
    if iseven(N)   # N is even
        S[2:div(N, 2)] = 2. * S[2:div(N, 2)]
        S[div(N, 2) + 2:end] = 0.
    else   # N is odd
        S[2:div(N + 1, 2)] = 2. * S[2:div(N + 1, 2)]
        S[div(N + 1, 2) + 1:end] = 0.
    end
    Ea = ifft(S)
    return Ea
end


"""Spectrum of real time signal -> spectrum of analytic time signal."""
function spectrum_real_to_analytic!(S::Array{Complex128, 1},
                                    Sa::Array{Complex128, 1})
    Nt = length(Sa)
    Sa[1] = S[1]
    if iseven(Nt)   # Nt is even
        @inbounds @views @. Sa[2:div(Nt, 2)] = 2. * S[2:div(Nt, 2)]
        Sa[div(Nt, 2) + 1] = S[div(Nt, 2) + 1]
    else   # Nt is odd
        @inbounds @views @. Sa[2:div(Nt + 1, 2)] = 2. * S[2:div(Nt + 1, 2)]
    end
    return nothing
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
