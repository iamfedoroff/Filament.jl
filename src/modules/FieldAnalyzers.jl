module FieldAnalyzers

import AnalyticSignals
import CUDA
import FFTW

import ..Fields
import ..Grids


abstract type FieldAnalyzer end


# ******************************************************************************
# R
# ******************************************************************************
mutable struct FieldAnalyzerR{
    T<:AbstractFloat,
    U<:AbstractArray{T},
    UG<:AbstractArray{T},
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Imax :: T
    rfil :: T
    P :: T
    # Storage arrays:
    rdr :: UG
    I :: U
    Igpu :: UG
end


function FieldAnalyzer(
    grid::Grids.GridR, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Imax, rfil, P = [zero(T) for i=1:3]

    rdr = @. grid.r * grid.dr
    rdr = CUDA.CuArray{T}(rdr)

    Nr = length(field.E)
    I = zeros(T, Nr)
    Igpu = CUDA.zeros(T, Nr)
    return FieldAnalyzerR(z, Imax, rfil, P, rdr, I, Igpu)
end


function analyze!(
    analyzer::FieldAnalyzerR, grid::Grids.GridR, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.Igpu = abs2(field.E)
    copyto!(analyzer.I, analyzer.Igpu)

    analyzer.Imax = maximum(analyzer.Igpu)

    analyzer.rfil = 2 * radius(grid.r, analyzer.I)

    analyzer.P = convert(T, 2 * pi) * sum(analyzer.Igpu .* analyzer.rdr)
    return nothing
end


# ******************************************************************************
# T
# ******************************************************************************
mutable struct FieldAnalyzerT{
    T<:AbstractFloat,
    U<:AbstractArray{T},
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Imax :: T
    rhomax :: T
    duration :: T
    F :: T
    # Storage arrays:
    I :: U
end


function FieldAnalyzer(
    grid::Grids.GridT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Imax, rhomax, duration, F = [zero(T) for i=1:4]
    Nt = length(field.E)
    I = zeros(T, Nt)
    return FieldAnalyzerT(z, Imax, rhomax, duration, F, I)
end


function analyze!(
    analyzer::FieldAnalyzerT, grid::Grids.GridT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.I = abs2(field.E)

    analyzer.Imax = maximum(analyzer.I)

    analyzer.rhomax = maximum(field.rho)

    analyzer.duration = radius(grid.t, analyzer.I)

    analyzer.F = sum(analyzer.I) * grid.dt
    return nothing
end


# ******************************************************************************
# RT
# ******************************************************************************
mutable struct FieldAnalyzerRT{
    T<:AbstractFloat,
    U1<:AbstractArray{T},
    U2<:AbstractArray{T},
    UG1<:AbstractArray{T},
    UG2<:AbstractArray{T},
    UCG2<:AbstractArray{Complex{T}},
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Fmax :: T
    Imax :: T
    rhomax :: T
    De :: T
    rfil :: T
    rpl :: T
    tau :: T
    W :: T
    # Storage arrays:
    rdr :: UG1
    Fr :: U2
    Ft :: U2
    rho :: U1
    S :: U2
    Frgpu :: UG2
    Ftgpu :: UG2
    rhogpu :: UG1
    Sgpu :: UG2
    Egpu :: UCG2
end


function FieldAnalyzer(
    grid::Grids.GridRT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Fmax, Imax, rhomax, De, rfil, rpl, tau, W = [zero(T) for i=1:8]

    rdr = @. grid.r * grid.dr
    rdr = CUDA.CuArray{T}(rdr)

    Nr, Nt = size(field.E)
    Nw = length(FFTW.rfftfreq(Nt))
    Fr = zeros(T, (Nr, 1))
    Ft = zeros(T, (1, Nt))
    rho = zeros(T, Nr)
    S = zeros(T, (1, Nw))
    Frgpu = CUDA.zeros(T, (Nr, 1))
    Ftgpu = CUDA.zeros(T, (1, Nt))
    rhogpu = CUDA.zeros(T, Nr)
    Sgpu = CUDA.zeros(T, (1, Nw))
    Egpu = CUDA.zeros(Complex{T}, (Nr, Nw))
    return FieldAnalyzerRT(
        z, Fmax, Imax, rhomax, De, rfil, rpl, tau, W,
        rdr, Fr, Ft, rho, S, Frgpu, Ftgpu, rhogpu, Sgpu, Egpu,
    )
end


function analyze!(
    analyzer::FieldAnalyzerRT, grid::Grids.GridRT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    analyzer.Frgpu .= sum(abs2.(field.E) .* grid.dt, dims=2)
    copyto!(analyzer.Fr, analyzer.Frgpu)

    analyzer.Ftgpu .= sum(
        convert(T, 2 * pi) .* abs2.(field.E) .* analyzer.rdr, dims=1,
    )
    copyto!(analyzer.Ft, analyzer.Ftgpu)

    @views @. analyzer.rhogpu = field.rho[:, end]
    copyto!(analyzer.rho, analyzer.rhogpu)

    # Integral power spectrum:
    #     Sa = ifft(Ea)
    #     Sr = aspec2rspec(Sa)
    #     Sr = 2 * Sr * Nt * dt
    #     S = 2 * pi * Int[|Sr|^2 * r * dr]
    FFTW.ldiv!(field.E, field.PT, field.E)   # time -> frequency [exp(-i*w*t)]
    AnalyticSignals.aspec2rspec!(analyzer.Egpu, field.E)
    analyzer.Sgpu .= convert(T, 2 * pi) *
                     sum(abs2.(2 * analyzer.Egpu * grid.Nt * grid.dt) .*
                         analyzer.rdr, dims=1)
    copyto!(analyzer.S, analyzer.Sgpu)
    FFTW.mul!(field.E, field.PT, field.E)   # frequency -> time [exp(-i*w*t)]

    analyzer.Fmax = maximum(analyzer.Frgpu)

    analyzer.Imax = maximum(abs2.(field.E))

    analyzer.rhomax = maximum(field.rho)

    analyzer.De = convert(T, 2 * pi) * sum(analyzer.rhogpu .* analyzer.rdr)

    analyzer.rfil = 2 * radius(grid.r, analyzer.Fr)

    analyzer.rpl = 2 * radius(grid.r, analyzer.rho)

    analyzer.tau = radius(grid.t, analyzer.Ft)

    analyzer.W = convert(T, 2 * pi) * sum(analyzer.Frgpu .* analyzer.rdr)
    return nothing
end


# ******************************************************************************
# XY
# ******************************************************************************
mutable struct FieldAnalyzerXY{
    T<:AbstractFloat,
    U<:AbstractArray{T},
    UG<:AbstractArray{T},
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Imax :: T
    ax :: T
    ay :: T
    P :: T
    # Storage arrays:
    I :: U
    Igpu :: UG
end


function FieldAnalyzer(
    grid::Grids.GridXY, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Imax, ax, ay, P = [zero(T) for i=1:4]
    Nx, Ny = size(field.E)
    I = zeros(T, (Nx, Ny))
    Igpu = CUDA.zeros(T, (Nx, Ny))

    return FieldAnalyzerXY(z, Imax, ax, ay, P, I, Igpu)
end


function analyze!(
    analyzer::FieldAnalyzerXY, grid::Grids.GridXY, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.Igpu = abs2(field.E)
    copyto!(analyzer.I, analyzer.Igpu)

    analyzer.Imax, imax = findmax(analyzer.I)

    @views analyzer.ax = radius(grid.x, analyzer.I[:, imax[2]])

    @views analyzer.ay = radius(grid.y, analyzer.I[imax[1], :])

    analyzer.P = sum(analyzer.Igpu) * grid.dx * grid.dy
    return nothing
end


# ******************************************************************************
# XYT
# ******************************************************************************
mutable struct FieldAnalyzerXYT{
    T<:AbstractFloat,
    U1<:AbstractArray{T},
    U2<:AbstractArray{T},
    UG1<:AbstractArray{T},
    UG2<:AbstractArray{T},
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Imax :: T
    rhomax :: T
    Fmax :: T
    ax :: T
    ay :: T
    W :: T
    # Storage arrays:
    Fxy :: U2
    Ft :: U1
    Fxygpu :: UG2
    Ftgpu :: UG1
end


function FieldAnalyzer(
    grid::Grids.GridXYT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Imax, rhomax, Fmax, ax, ay, W = [zero(T) for i=1:6]
    Fxy = zeros(T, (grid.Nx, grid.Ny, 1))
    Ft = zeros(T, (1, 1, grid.Nt))
    Fxygpu = CUDA.zeros(T, (grid.Nx, grid.Ny, 1))
    Ftgpu = CUDA.zeros(T, (1, 1, grid.Nt))
    return FieldAnalyzerXYT(
        z, Imax, rhomax, Fmax, ax, ay, W,  Fxy, Ft, Fxygpu, Ftgpu,
    )
end


function analyze!(
    analyzer::FieldAnalyzerXYT, grid::Grids.GridXYT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    analyzer.Imax = maximum(abs2.(field.E))
    analyzer.rhomax = maximum(field.rho)

    analyzer.Fxygpu .= sum(abs2.(field.E) .* grid.dt, dims=3)
    copyto!(analyzer.Fxy, analyzer.Fxygpu)

    analyzer.Ftgpu .= sum(abs2.(field.E) .* grid.dx .* grid.dy, dims=[1, 2])
    copyto!(analyzer.Ft, analyzer.Ftgpu)

    analyzer.Fmax, imax = findmax(analyzer.Fxy)

    @views analyzer.ax = radius(grid.x, analyzer.Fxy[:, imax[2], 1])
    @views analyzer.ay = radius(grid.y, analyzer.Fxy[imax[1], :, 1])

    analyzer.W = sum(analyzer.Ftgpu) * grid.dt
    return nothing
end


# ******************************************************************************
function radius(
    x::AbstractArray{T}, y::AbstractArray{T}, level::T=convert(T, exp(-1)),
) where T<:AbstractFloat
    Nx = length(x)
    ylevel = maximum(y) * level

    radl = zero(T)
    for i=1:Nx
        if y[i] >= ylevel
            radl = x[i]
            break
        end
    end

    radr = zero(T)
    for i=Nx:-1:1
        if y[i] >= ylevel
            radr = x[i]
            break
        end
    end

    return (abs(radl) + abs(radr)) / 2
end


end
