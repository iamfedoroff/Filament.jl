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

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put
    # H(t) into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * unit.t
    Hraman = @. Hraman + 0im   # real -> complex

    # The correct way to calculate spectrum which matches theory:
    #    S = ifftshift(E)   # compensation of the spectrum oscillations
    #    S = ifft(S) * len(E) * dt   # normalization
    #    S = fftshift(S)   # recovery of the proper array order
    Hraman = FFTW.ifftshift(Hraman)
    FFTW.ifft!(Hraman)   # time -> frequency [exp(-i*w*t)]
    @. Hraman = Hraman * grid.Nt * grid.dt

    if ! isa(grid, Grids.GridT)   # FIXME: should be removed in a generic code.
        Rnl = convert(FloatGPU, Rnl)
        Hraman = CUDA.CuArray{Complex{FloatGPU}}(Hraman)
    end

    if THG
        calc = calc_raman
    else
        calc = calc_raman_nothg
    end

    p = (Hraman, field.PT)
    return Media.NonlinearResponse(Rnl, calc, p)
end


function calc_raman(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    Hraman, PT = p
    @. F = real(E)^2
    convolution!(F, PT, Hraman)
    @. F = F * real(E)
    return nothing
end


function calc_raman_nothg(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    Hraman, PT = p
    @. F = 3 / 4 * abs2(E)
    convolution!(F, PT, Hraman)
    @. F = F * real(E)
    return nothing
end


function convolution!(
    x::AbstractArray{Complex{T}, 1},
    plan::FFTW.Plan,
    H::AbstractArray{Complex{T}, 1},
) where T
    plan \ x   # time -> frequency [exp(-i*w*t)]
    @. x = H * x
    plan * x   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function convolution!(
    x::Union{CUDA.CuArray{Complex{T}, 2}, CUDA.CuArray{Complex{T}, 3}},
    plan::FFTW.Plan,
    H::CUDA.CuArray{Complex{T}, 1},
) where T
    plan \ x   # time -> frequency [exp(-i*w*t)]

    N = length(x)

    ckernel = CUDA.@cuda launch=false _convolution_kernel!(x, H)
    config = CUDA.launch_configuration(ckernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    ckernel(x, H; threads=threads, blocks=blocks)

    plan * x   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function _convolution_kernel!(x, H)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    nt = ndims(x)
    cartesian = CartesianIndices(size(x))
    for I=id:stride:length(x)
        it = cartesian[I][nt]
        x[I] = H[it] * x[I]
    end
    return nothing
end
