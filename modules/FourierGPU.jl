module FourierGPU

import CuArrays

using PyCall
@pyimport numpy.fft as npfft

const FloatGPU = Float32
const ComplexGPU = Complex64


struct FourierTransform
    Nt :: Int64
    Nw :: Int64
    HS_gpu :: CuArrays.CuArray{FloatGPU, 1}
    Ec_gpu :: CuArrays.CuArray{ComplexGPU, 1}
    Sc_gpu :: CuArrays.CuArray{ComplexGPU, 1}
    Sr_gpu :: CuArrays.CuArray{ComplexGPU, 1}
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
    HS_gpu = CuArrays.CuArray(convert(Array{FloatGPU, 1}, HS))

    # arrays to store intermediate results:
    Ec_gpu = CuArrays.CuArray(zeros(ComplexGPU, Nt))
    Sc_gpu = CuArrays.CuArray(zeros(ComplexGPU, Nt))
    Sr_gpu = CuArrays.CuArray(zeros(ComplexGPU, Nw))

    CuArrays.allowscalar(false)   # disable slow fallback methods

    PFFT = plan_fft(CuArrays.CuArray(zeros(ComplexGPU, Nt)))
    PIFFT = plan_ifft(CuArrays.CuArray(zeros(ComplexGPU, Nt)))
    PRFFT = plan_rfft(CuArrays.CuArray(zeros(FloatGPU, Nt)))
    PIRFFT = plan_irfft(CuArrays.CuArray(zeros(ComplexGPU, Nw)), Nt)

    # A_mul_B!(Sc_gpu, PFFT, Ec_gpu)
    # A_mul_B!(Ec_gpu, PIFFT, Sc_gpu)
    # A_mul_B!(Sr_gpu, PRFFT, Er_gpu)
    # A_mul_B!(Er_gpu, PIRFFT, Sr_gpu)

    return FourierTransform(Nt, Nw, HS_gpu, Ec_gpu, Sc_gpu, Sr_gpu,
                            PFFT, PIFFT, PRFFT, PIRFFT)
end


function fft1d!(FT::FourierTransform, Ec_gpu::CuArrays.CuArray{ComplexGPU, 1},
                Sc_gpu::CuArrays.CuArray{ComplexGPU, 1})
    A_mul_B!(Sc_gpu, FT.PFFT, Ec_gpu)
    return nothing
end


function fft1d(FT::FourierTransform, Ec_gpu::CuArrays.CuArray{ComplexGPU, 1})
    Sc_gpu = CuArrays.CuArray(zeros(ComplexGPU, FT.Nt))
    fft1d!(FT, Ec_gpu, Sc_gpu)   # time -> frequency
    return Sc_gpu
end


function ifft1d!(FT::FourierTransform, Sc_gpu::CuArrays.CuArray{ComplexGPU, 1},
                 Ec_gpu::CuArrays.CuArray{ComplexGPU, 1})
    A_mul_B!(Ec_gpu, FT.PIFFT, Sc_gpu)
    return nothing
end


function ifft1d(FT::FourierTransform, Sc_gpu::CuArrays.CuArray{ComplexGPU, 1})
    Ec_gpu = CuArrays.CuArray(zeros(ComplexGPU, FT.Nt))
    ifft1d!(FT, Sc_gpu, Ec_gpu)   # frequency -> time
    return Ec_gpu
end


function rfft1d!(FT::FourierTransform, Er_gpu::CuArrays.CuArray{FloatGPU, 1},
                 Sr_gpu::CuArrays.CuArray{ComplexGPU, 1})
    A_mul_B!(Sr_gpu, FT.PRFFT, Er_gpu)   # time -> frequency
    return nothing
end


function rfft1d(FT::FourierTransform, Er_gpu::CuArrays.CuArray{FloatGPU, 1})
    Sr_gpu = CuArrays.CuArray(zeros(ComplexGPU, FT.Nw))
    rfft1d!(FT, Er_gpu, Sr_gpu)   # time -> frequency
    return Sr_gpu
end


function rfft2d!(FT::FourierTransform, E_gpu::CuArrays.CuArray{FloatGPU, 2},
                 S_gpu::CuArrays.CuArray{ComplexGPU, 2})
    Nr = size(E_gpu, 1)
    for i=1:Nr
        Er_gpu = E_gpu[i, :]
        rfft1d!(FT, Er_gpu, FT.Sr_gpu)   # time -> frequency
        @inbounds S_gpu[i, :] = FT.Sr_gpu
    end
    return nothing
end


function irfft1d!(FT::FourierTransform, Sr_gpu::CuArrays.CuArray{ComplexGPU, 1},
                  Er_gpu::CuArrays.CuArray{FloatGPU, 1})
    A_mul_B!(Er_gpu, FT.PIRFFT, Sr_gpu)   # frequency -> time
    return nothing
end


function irfft1d(FT::FourierTransform, Sr_gpu::CuArrays.CuArray{ComplexGPU, 1})
    Er_gpu = CuArrays.CuArray(zeros(Float64, FT.Nt))
    irfft1d!(FT, Sr_gpu, Er_gpu)   # frequency -> time
    return Er_gpu
end


"""Real time signal -> analytic time signal."""
function signal_real_to_signal_analytic(FT::FourierTransform,
                                        Er_gpu::CuArrays.CuArray{FloatGPU, 1})
    # Need test for odd N and low frequencies
    S_gpu = fft1d(FT, Er_gpu)
    @. S_gpu = FT.HS_gpu * S_gpu
    Ec_gpu = ifft1d(FT, S_gpu)
    return Ec_gpu
end


"""Spectrum of real time signal -> analytic time signal."""
function spectrum_real_to_signal_analytic!(FT::FourierTransform,
                                           Sr_gpu::CuArrays.CuArray{ComplexGPU, 1},
                                           Ec_gpu::CuArrays.CuArray{ComplexGPU, 1})
    # Need test for odd N and low frequencies
    # S = vcat(Sr, conj(Sr[end-1:-1:2]))
    @inbounds FT.Sc_gpu[1:FT.Nw] = Sr_gpu
    @inbounds @. FT.Sc_gpu = FT.HS_gpu * FT.Sc_gpu
    ifft1d!(FT, FT.Sc_gpu, Ec_gpu)
    return nothing
end


function spectrum_real_to_signal_analytic_2d!(FT::FourierTransform,
                                             S_gpu::CuArrays.CuArray{ComplexGPU, 2},
                                             E_gpu::CuArrays.CuArray{ComplexGPU, 2})
    Nr = size(E_gpu, 1)
    for i=1:Nr
        @inbounds Sr_gpu = S_gpu[i, :]
        spectrum_real_to_signal_analytic!(FT, Sr_gpu, FT.Ec_gpu)
        @inbounds E_gpu[i, :] = FT.Ec_gpu
    end
end


function convolution!(FT::FourierTransform,
                      Hw_gpu::CuArrays.CuArray{ComplexGPU, 1},
                      x_gpu::CuArrays.CuArray{FloatGPU, 1},
                      res_gpu::CuArrays.CuArray{FloatGPU, 1})
    rfft1d!(FT, x_gpu, FT.Sr_gpu)
    @inbounds @. FT.Sr_gpu = Hw_gpu * FT.Sr_gpu
    irfft1d!(FT, FT.Sr_gpu, res_gpu)
    return nothing
end


end
