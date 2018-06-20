module Fourier


function fft1d(E::Array{Complex128, 1})
    S = ifft(E) * length(E)   # time -> frequency
    return S
end


function ifft1d(S::Array{Complex128, 1})
    E = fft(S) / length(S)   # frequency -> time
    return E
end


function rfft1d(E::Array{Float64, 1})
    S = conj(rfft(E))   # time -> frequency
    return S
end


function irfft1d(S::Array{Complex128, 1})
    E = irfft(conj(S))   # frequency -> time
    return E
end


"""Real time signal -> analytic time signal."""
function signal_real_to_analytic(Er::Array{Float64, 1})
    N = length(Er)
    S = fft1d(Er + 0im)   # time -> frequency
    if N % 2 == 0   # N is even
        S[2:div(N, 2)] = 2. * S[2:div(N, 2)]
        S[div(N, 2) + 2:end] = 0.
    else:   # N is odd
        S[2:div(N + 1, 2)] = 2. * S[2:div(N + 1, 2)]
        S[div(N + 1, 2) + 1:end] = 0.
    end
    Ea = ifft1d(S)
    return Ea
end


"""Spectrum of real time signal -> spectrum of analytic time signal."""
function spectrum_real_to_analytic(S::Array{Complex128, 1}, Nt::Int64)
    Sa = zeros(Complex128, Nt)
    Sa[1] = S[1]
    if Nt % 2 == 0   # Nt is even
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
