module FourierTransforms

import FFTW
import CUDA


struct Plan{P<:FFTW.Plan, PI<:FFTW.Plan}
    pfft :: P
    pifft :: PI
end


function Plan(F::AbstractArray{Complex{T}}, region=nothing) where T
    if region == nothing
        region = [i for i=1:ndims(F)]
    end
    pfft = FFTW.plan_fft!(F, region)
    pifft = FFTW.plan_ifft!(F, region)
    # in-place FFTs results in segfault after run completion
    # https://github.com/JuliaGPU/CUDA.jl/issues/95
    # pfft = FFTW.plan_fft!(F, region)
    # pifft = FFTW.plan_ifft!(F, region)
    return Plan(pfft, pifft)
end


function Plan(F::CUDA.CuArray{Complex{T}}, region=nothing) where T
    if region == nothing
        region = [i for i=1:ndims(F)]
    end
    pfft = FFTW.plan_fft(F, region)
    pifft = FFTW.plan_ifft(F, region)
    # in-place FFTs results in segfault after run completion
    # https://github.com/JuliaGPU/CUDA.jl/issues/95
    # pfft = FFTW.plan_fft!(F, region)
    # pifft = FFTW.plan_ifft!(F, region)
    return Plan(pfft, pifft)
end


function fft!(E::AbstractArray{Complex{T}}, plan::Plan) where T
    FFTW.mul!(E, plan.pfft, E)
    # plan.pfft * E   # results in segfault after run completion
    return nothing
end


function ifft!(E::AbstractArray{Complex{T}}, plan::Plan) where T
    FFTW.mul!(E, plan.pifft, E)
    # plan.pifft * E   # results in segfault after run completion
    return nothing
end


function convolution!(
    x::AbstractArray{Complex{T}, 1},
    plan::Plan,
    H::AbstractArray{Complex{T}, 1},
) where T
    ifft!(x, plan)   # time -> frequency [exp(-i*w*t)]
    @. x = H * x
    fft!(x, plan)   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function convolution!(
    x::CUDA.CuArray{Complex{T}, 2},
    plan::Plan,
    H::CUDA.CuArray{Complex{T}, 1},
) where T
    ifft!(x, plan)   # time -> frequency [exp(-i*w*t)]

    N = length(x)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config _convolution_kernel!(x, H)

    fft!(x, plan)   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function _convolution_kernel!(x, H)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nr, Nt = size(x)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        x[i, j] = H[j] * x[i, j]
    end
    return nothing
end


function rfft_length(Nt::Int)
    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end
    return Nw
end


end
