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


"""
Return the Discrete Fourier Transform sample frequencies.
https://github.com/numpy/numpy/blob/v1.15.0/numpy/fft/helper.py#L124-L169
"""
function fftfreq(n::Int, d::T) where T<:AbstractFloat
    val = 1 / (n * d)
    results = zeros(T, n)
    N = Int(floor((n - 1) / 2)) + 1
    p1 = Array(0:N-1)
    results[1:N] = p1
    p2 = Array(-Int(floor(n / 2)):-1)
    results[N+1:end] = p2
    return results * val
end


"""
Return the Discrete Fourier Transform sample frequencies (for usage with
rfft, irfft).
https://github.com/numpy/numpy/blob/v1.15.0/numpy/fft/helper.py#L173-L221
"""
function rfftfreq(n::Int, d::T) where T<:AbstractFloat
    val = 1 / (n * d)
    N = Int(floor(n / 2))
    results = Array{T}(0:N)
    return results * val
end


function rfft_length(Nt::Int)
    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end
    return Nw
end


"""
Circular-shift along the given dimension of a periodic signal 'x' centered at
index '1' so it becomes centered at index 'N / 2 + 1', where 'N' is the size of
that dimension.
https://github.com/JuliaMath/AbstractFFTs.jl/blob/66695a72b2a29a059a9bf5fae51a3107172f146d/src/definitions.jl#L349
"""
fftshift(x) = circshift(x, div.([size(x)...],2))


"""
Circular-shift along the given dimension of a periodic signal 'x' centered at
index 'N / 2 + 1' so it becomes centered at index '1', where 'N' is the size of
that dimension.
https://github.com/JuliaMath/AbstractFFTs.jl/blob/66695a72b2a29a059a9bf5fae51a3107172f146d/src/definitions.jl#L374
"""
ifftshift(x) = circshift(x, div.([size(x)...],-2))


end
