module FieldAnalyzers

import CuArrays

import Fields
import Grids


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
    I :: U
    Igpu :: UG
end


function FieldAnalyzer(field::Fields.FieldR, z::T) where T<:AbstractFloat
    Imax, rfil, P = [zero(T) for i=1:3]
    Nr = length(field.E)
    I = zeros(T, Nr)
    Igpu = CuArrays.zeros(T, Nr)
    return FieldAnalyzerR(z, Imax, rfil, P, I, Igpu)
end


function analyze!(
    analyzer::FieldAnalyzerR, grid::Grids.GridR, field::Fields.FieldR, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.Igpu = abs2(field.E)
    copyto!(analyzer.I, analyzer.Igpu)

    analyzer.Imax = maximum(analyzer.Igpu)

    analyzer.rfil = 2 * Grids.radius(grid.r, analyzer.I)

    # Warning! Igpu is rewritten!
    @. analyzer.Igpu = analyzer.Igpu * grid.rdr
    analyzer.P = convert(T, 2 * pi) * sum(analyzer.Igpu)
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


function FieldAnalyzer(field::Fields.FieldT, z::T) where T<:AbstractFloat
    Imax, rhomax, duration, F = [zero(T) for i=1:4]
    Nt = length(field.E)
    I = zeros(T, Nt)
    return FieldAnalyzerT(z, Imax, rhomax, duration, F, I)
end


function analyze!(
    analyzer::FieldAnalyzerT, grid::Grids.GridT, field::Fields.FieldT, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.I = abs2(field.E)

    analyzer.Imax = maximum(analyzer.I)

    analyzer.rhomax = maximum(field.rho)

    analyzer.duration = Grids.radius(grid.t, analyzer.I)

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
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Fmax :: T
    Imax :: T
    rhomax :: T
    De :: T
    rfil :: T
    rpl :: T
    W :: T
    # Storage arrays:
    F :: U2
    rho :: U1
    S :: U2
    Fgpu :: UG2
    rhogpu :: UG1
    Sgpu :: UG2
    Igpu :: UG2
end


function FieldAnalyzer(field::Fields.FieldRT, z::T) where T<:AbstractFloat
    Fmax, Imax, rhomax, De, rfil, rpl, W = [zero(T) for i=1:7]
    Nr, Nt = size(field.E)
    Nr, Nw = size(field.S)
    F = zeros(T, (Nr, 1))
    rho = zeros(T, Nr)
    S = zeros(T, (1, Nw))
    Fgpu = CuArrays.zeros(T, (Nr, 1))
    rhogpu = CuArrays.zeros(T, Nr)
    Sgpu = CuArrays.zeros(T, (1, Nw))
    Igpu = CuArrays.zeros(T, (Nr, Nt))
    return FieldAnalyzerRT(
        z, Fmax, Imax, rhomax, De, rfil, rpl, W, F, rho, S, Fgpu, rhogpu, Sgpu, Igpu,
    )
end


function analyze!(
    analyzer::FieldAnalyzerRT, grid::Grids.GridRT, field::Fields.FieldRT, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.Igpu = abs2(field.E)

    analyzer.Fgpu .= sum(analyzer.Igpu .* grid.dt, dims=2)
    copyto!(analyzer.F, analyzer.Fgpu)

    @views @. analyzer.rhogpu = field.rho[:, end]
    copyto!(analyzer.rho, analyzer.rhogpu)

    # Integral power spectrum:
    #     Ew = rfft(Et)
    #     Ew = 2 * Ew * dt
    #     S = 2 * pi * Int[|Ew|^2 * r * dr]
    analyzer.Sgpu .= sum(convert(T, 8 * pi) .* abs2.(field.S) .* grid.rdr .*
                         grid.dt^2, dims=1)
    copyto!(analyzer.S, analyzer.Sgpu)

    analyzer.Fmax = maximum(analyzer.Fgpu)

    analyzer.Imax = maximum(analyzer.Igpu)

    analyzer.rhomax = maximum(analyzer.rhogpu)

    analyzer.De = convert(T, 2 * pi) * sum(analyzer.rhogpu .* grid.rdr)

    analyzer.rfil = 2 * Grids.radius(grid.r, analyzer.F)

    analyzer.rpl = 2 * Grids.radius(grid.r, analyzer.rho)

    analyzer.W = convert(T, 2 * pi) * sum(analyzer.Fgpu .* grid.rdr)
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


function FieldAnalyzer(field::Fields.FieldXY, z::T) where T<:AbstractFloat
    Imax, ax, ay, P = [zero(T) for i=1:4]
    Nx, Ny = size(field.E)
    I = zeros(T, (Nx, Ny))
    Igpu = CuArrays.zeros(T, (Nx, Ny))

    return FieldAnalyzerXY(z, Imax, ax, ay, P, I, Igpu)
end


function analyze!(
    analyzer::FieldAnalyzerXY, grid::Grids.GridXY, field::Fields.FieldXY, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    @. analyzer.Igpu = abs2(field.E)
    copyto!(analyzer.I, analyzer.Igpu)

    analyzer.Imax, imax = findmax(analyzer.I)

    @views analyzer.ax = Grids.radius(grid.x, analyzer.I[:, imax[2]])

    @views analyzer.ay = Grids.radius(grid.y, analyzer.I[imax[1], :])

    analyzer.P = sum(analyzer.Igpu) * grid.dx * grid.dy
    return nothing
end


end
