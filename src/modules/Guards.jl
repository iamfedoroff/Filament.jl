module Guards

import CUDA

import ..Fields
import ..Grids
import ..Media
import ..Units


abstract type Guard end


struct Guard1D{A<:AbstractArray} <: Guard
    F :: A   # field filter
    S :: A   # spectrum filter
end


struct Guard2D{A<:AbstractArray, A2<:AbstractArray} <: Guard
    F1 :: A   # field filter along dimension 1
    F2 :: A   # field filter along dimension 2
    S1 :: A2   # spectrum filter along dimension 1 or (1, 2)
    S2 :: A   # spectrum filter along dimension 2
end


struct Guard3D{A<:AbstractArray, A2<:AbstractArray} <: Guard
    F1 :: A   # field filter along dimension 1
    F2 :: A   # field filter along dimension 2
    F3 :: A   # field filter along dimension 3
    S1 :: A2   # spectrum filter along dimension 1
    S2 :: A2   # spectrum filter along dimension 2
    S3 :: A   # spectrum filter along dimension 3
end


# ******************************************************************************
# R
# ******************************************************************************
function Guard(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.Field,
    medium::Media.Medium,
    rguard::T,
    kguard::T,
) where T<:AbstractFloat
    Rguard = guard_window_right(grid.r, rguard)

    k0 = Media.k_func(medium, field.w0)
    kmax = k0 * sind(kguard)
    Kguard = @. exp(-((grid.k * unit.k)^2 / kmax^2)^20)

    Rguard = CUDA.CuArray{T}(Rguard)
    Kguard = CUDA.CuArray{T}(Kguard)
    return Guard1D(Rguard, Kguard)
end


function Guard(
    unit::Units.UnitR,
    grid::Grids.GridRn,
    field::Fields.Field,
    medium::Media.Medium,
    rguard::T,
    kguard::T,
) where T<:AbstractFloat
    Rguard = guard_window_right(grid.r, rguard)
    Kguard = ones(grid.Nr)
    return Guard1D(Rguard, Kguard)
end


# ******************************************************************************
# T
# ******************************************************************************
function Guard(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.Field,
    medium::Media.Medium,
    tguard::T,
    wguard::T,
) where T<:AbstractFloat
    Tguard = guard_window_both(grid.t, tguard)
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)
    return Guard1D(Tguard, Wguard)
end


# ******************************************************************************
# RT
# ******************************************************************************
function Guard(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.Field,
    medium::Media.Medium,
    rguard::T,
    tguard::T,
    kguard::T,
    wguard::T,
) where T<:AbstractFloat
    Rguard = guard_window_right(grid.r, rguard)
    Tguard = guard_window_both(grid.t, tguard)

    Kguard = zeros((grid.Nr, grid.Nt))
    for j=1:grid.Nt
        k = Media.k_func(medium, grid.w[j] * unit.w)
        kmax = k * sind(kguard)
        if kmax != 0
            for i=1:grid.Nr
                Kguard[i, j] = exp(-((grid.k[i] * unit.k)^2 / kmax^2)^20)
            end
        end
    end

    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)

    Rguard = CUDA.CuArray{T}(Rguard)
    Tguard = CUDA.CuArray{T}(Tguard)
    Kguard = CUDA.CuArray{T}(Kguard)
    Wguard = CUDA.CuArray{T}(Wguard)
    return Guard2D(Rguard, Tguard, Kguard, Wguard)
end


# ******************************************************************************
# XY
# ******************************************************************************
function Guard(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.Field,
    medium::Media.Medium,
    xguard::T,
    yguard::T,
    kxguard::T,
    kyguard::T,
) where T<:AbstractFloat
    Xguard = guard_window_both(grid.x, xguard)
    Yguard = guard_window_both(grid.y, yguard)

    k0 = Media.k_func(medium, field.w0)
    kxmax = k0 * sind(kxguard)
    kymax = k0 * sind(kyguard)
    KXguard = @. exp(-((grid.kx * unit.kx)^2 / kxmax^2)^20)
    KYguard = @. exp(-((grid.ky * unit.ky)^2 / kymax^2)^20)

    Xguard = CUDA.CuArray{T}(Xguard)
    Yguard = CUDA.CuArray{T}(Yguard)
    KXguard = CUDA.CuArray{T}(KXguard)
    KYguard = CUDA.CuArray{T}(KYguard)
    return Guard2D(Xguard, Yguard, KXguard, KYguard)
end


# ******************************************************************************
# XYT
# ******************************************************************************
function Guard(
    unit::Units.UnitXYT,
    grid::Grids.GridXYT,
    field::Fields.Field,
    medium::Media.Medium,
    xguard::T,
    yguard::T,
    tguard::T,
    kxguard::T,
    kyguard::T,
    wguard::T,
) where T<:AbstractFloat
    Xguard = guard_window_both(grid.x, xguard)
    Yguard = guard_window_both(grid.y, yguard)
    Tguard = guard_window_both(grid.t, tguard)

    # k0 = Media.k_func(medium, field.w0)
    # kxmax = k0 * sind(kxguard)
    # kymax = k0 * sind(kyguard)
    # KXguard = @. exp(-((grid.kx * unit.kx)^2 / kxmax^2)^20)
    # KYguard = @. exp(-((grid.ky * unit.ky)^2 / kymax^2)^20)
    KXguard = ones(length(grid.kx))
    KYguard = ones(length(grid.ky))

    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)

    Xguard = CUDA.CuArray{T}(Xguard)
    Yguard = CUDA.CuArray{T}(Yguard)
    Tguard = CUDA.CuArray{T}(Tguard)
    KXguard = CUDA.CuArray{T}(KXguard)
    KYguard = CUDA.CuArray{T}(KYguard)
    Wguard = CUDA.CuArray{T}(Wguard)
    return Guard3D(Xguard, Yguard, Tguard, KXguard, KYguard, Wguard)
end


# ******************************************************************************
function apply_field_filter!(E::AbstractArray{T, 1}, guard::Guard1D) where T
    @. E = E * guard.F
end


function apply_field_filter!(E::CUDA.CuArray{T, 2}, guard::Guard2D) where T
    N = length(E)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config kernel!(E, guard.F1, guard.F2)
end


function apply_field_filter!(E::CUDA.CuArray{T, 3}, guard::Guard3D) where T
    N = length(E)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config kernel!(E, guard.F1, guard.F2, guard.F3)
end


# ******************************************************************************
function apply_spectral_filter!(E::AbstractArray{T, 1}, guard::Guard1D) where T
    @. E = E * guard.S
end


function apply_spectral_filter!(E::CUDA.CuArray{T, 2}, guard::Guard2D) where T
    N = length(E)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config kernel!(E, guard.S1, guard.S2)
end


function apply_spectral_filter!(E::CUDA.CuArray{T, 3}, guard::Guard3D) where T
    N = length(E)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config kernel!(E, guard.S1, guard.S2, guard.S3)
end


# ******************************************************************************
function kernel!(
    F::CUDA.CuDeviceArray{Complex{T}, 2},
    A::CUDA.CuDeviceArray{T, 1},
    B::CUDA.CuDeviceArray{T, 1},
) where T<:AbstractFloat
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    N1, N2 = size(F)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        F[i, j] = F[i, j] * A[i] * B[j]
    end
    return nothing
end


function kernel!(
    F::CUDA.CuDeviceArray{Complex{T}, 2},
    A::CUDA.CuDeviceArray{T, 2},
    B::CUDA.CuDeviceArray{T, 1},
) where T<:AbstractFloat
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    N1, N2 = size(F)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        F[i, j] = F[i, j] * A[i, j] * B[j]
    end
    return nothing
end


function kernel!(
    F::CUDA.CuDeviceArray{Complex{T}, 3},
    A::CUDA.CuDeviceArray{T, 1},
    B::CUDA.CuDeviceArray{T, 1},
    C::CUDA.CuDeviceArray{T, 1},
) where T<:AbstractFloat
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    N1, N2, N3 = size(F)
    cartesian = CartesianIndices((N1, N2, N3))
    for k=id:stride:N1*N2*N3
        i1 = cartesian[k][1]
        i2 = cartesian[k][2]
        i3 = cartesian[k][2]
        F[i1, i2, i3] = F[i1, i2, i3] * A[i1] * B[i2] * C[i3]
    end
    return nothing
end


# ******************************************************************************
function guard_window_left(
    x::AbstractArray{T, 1}, width::T; p::Int=6,
) where T<:AbstractFloat
    if width >= (x[end] - x[1])
        error("Guard width is larger or equal than the grid size.")
    end
    N = length(x)
    if width == 0
        guard = ones(T, N)
    else
        xloc1 = x[1]
        xloc2 = x[1] + width
        gauss1 = zeros(T, N)
        gauss2 = ones(T, N)
        for i=1:N
            if x[i] >= xloc1
                gauss1[i] = 1 - exp(-((x[i] - xloc1) / (width / 2))^p)
            end
            if x[i] <= xloc2
                gauss2[i] = exp(-((x[i] - xloc2) / (width / 2))^p)
            end
        end
        guard = @. (gauss1 + gauss2) / 2
    end
    return guard
end


function guard_window_right(
    x::AbstractArray{T, 1}, width::T; p::Int=6,
) where T<:AbstractFloat
    if width >= (x[end] - x[1])
        error("Guard width is larger or equal than the grid size.")
    end
    N = length(x)
    if width == 0
        guard = ones(T, N)
    else
        xloc1 = x[end] - width
        xloc2 = x[end]
        gauss1 = ones(T, N)
        gauss2 = zeros(T, N)
        for i=1:N
            if x[i] >= xloc1
                gauss1[i] = exp(-((x[i] - xloc1) / (width / 2))^p)
            end
            if x[i] <= xloc2
                gauss2[i] = 1 - exp(-((x[i] - xloc2) / (width / 2))^p)
            end
        end
        guard = @. (gauss1 + gauss2) / 2
    end
    return guard
end


function guard_window_both(
    x::AbstractArray{T, 1}, width::T; p::Int=6,
) where T<:AbstractFloat
    if width >= (x[end] - x[1]) / 2
        error("Guard width is larger or equal than the grid size.")
    end
    lguard = guard_window_left(x, width; p=p)
    rguard = guard_window_right(x, width; p=p)
    return @. lguard + rguard - 1
end


end
