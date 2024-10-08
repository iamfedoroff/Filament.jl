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
    grid::Union{Grids.GridR, Grids.GridRn}, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Imax, rfil, P = [zero(T) for i=1:3]

    rdr = @. grid.r * grid.dr

    Nr = length(field.E)
    I = zeros(T, Nr)

    if typeof(grid) <: Grids.GridR
        rdr = CUDA.CuArray{T}(rdr)
        Igpu = CUDA.zeros(T, Nr)
    elseif typeof(grid) <: Grids.GridRn
        Igpu = zeros(T, Nr)
    end
    return FieldAnalyzerR(z, Imax, rfil, P, rdr, I, Igpu)
end


function analyze!(
    analyzer::FieldAnalyzerR, grid::Union{Grids.GridR, Grids.GridRn}, field::Fields.Field, z::T,
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
    A<:AbstractArray{T},
    AG<:AbstractArray{T},
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
    Fr :: A
    Ft :: A
    rho :: A
    S :: A
    rdr :: AG
    rhogpu :: AG
end


function FieldAnalyzer(
    grid::Grids.GridRT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Fmax, Imax, rhomax, De, rfil, rpl, tau, W = [zero(T) for i=1:8]

    rdr = @. grid.r * grid.dr
    rdr = CUDA.CuArray{T}(rdr)

    Nr, Nt = size(field.E)
    Nw = length(FFTW.rfftfreq(Nt))
    Fr = zeros(T, Nr)
    Ft = zeros(T, Nt)
    rho = zeros(T, Nr)
    S = zeros(T, Nw)
    rhogpu = CUDA.zeros(T, Nr)
    return FieldAnalyzerRT(
        z, Fmax, Imax, rhomax, De, rfil, rpl, tau, W, Fr, Ft, rho, S, rdr, rhogpu,
    )
end


function analyze!(
    analyzer::FieldAnalyzerRT, grid::Grids.GridRT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    Frgpu = vec(sum(abs2, field.E, dims=2)) * grid.dt
    copyto!(analyzer.Fr, Frgpu)

    Ftgpu = convert(T, 2 * pi) *
            vec(sum(abs2.(field.E) .* analyzer.rdr, dims=1))
    copyto!(analyzer.Ft, Ftgpu)

    @views @. analyzer.rhogpu = field.rho[:, end]
    copyto!(analyzer.rho, analyzer.rhogpu)

    # Integral power spectrum:
    #     Sa = ifft(Ea)
    #     Sr = aspec2rspec(Sa)
    #     Sr = 2 * Sr * Nt * dt
    #     S = 2 * pi * Int[|Sr|^2 * r * dr]
    Nw = length(FFTW.rfftfreq(grid.Nt))
    field.PT \ field.E   # time -> frequency [exp(-i*w*t)]
    AnalyticSignals.aspec2rspec!(field.E)
    Sgpu = @views convert(T, 2 * pi) * vec(
        sum(
            abs2.(2 * field.E[:,1:Nw] * grid.Nt * grid.dt) .* analyzer.rdr;
            dims=1
        )
    )
    copyto!(analyzer.S, Sgpu)
    AnalyticSignals.rspec2aspec!(field.E)
    field.PT * field.E   # frequency -> time [exp(-i*w*t)]

    analyzer.Fmax = maximum(Frgpu)

    analyzer.Imax = maximum(abs2, field.E)

    analyzer.rhomax = maximum(field.rho)

    analyzer.De = convert(T, 2 * pi) * sum(analyzer.rhogpu .* analyzer.rdr)

    analyzer.rfil = 2 * radius(grid.r, analyzer.Fr)

    analyzer.rpl = 2 * radius(grid.r, analyzer.rho)

    analyzer.tau = radius(grid.t, analyzer.Ft)

    analyzer.W = convert(T, 2 * pi) * sum(Frgpu .* analyzer.rdr)
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
    U3<:AbstractArray{T},
} <: FieldAnalyzer
    # Observed variables:
    z :: T
    Fmax :: T
    Imax :: T
    rhomax :: T
    De :: T
    ax :: T
    ay :: T
    W :: T
    # Storage arrays:
    Fxy :: U2
    Ft :: U1
    rho :: U2
    S :: U3
end


function FieldAnalyzer(
    grid::Grids.GridXYT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    Fmax, Imax, rhomax, De, ax, ay, W = [zero(T) for i=1:7]
    Fxy = zeros(T, (grid.Nx, grid.Ny))
    Ft = zeros(T, grid.Nt)
    rho = zeros(T, (grid.Nx, grid.Ny))
    Nw = length(FFTW.rfftfreq(grid.Nt))
    S = zeros(T, Nw)
    return FieldAnalyzerXYT(z, Fmax, Imax, rhomax, De, ax, ay, W, Fxy, Ft, rho, S)
end


function analyze!(
    analyzer::FieldAnalyzerXYT, grid::Grids.GridXYT, field::Fields.Field, z::T,
) where T<:AbstractFloat
    analyzer.z = z

    Fxygpu = dropdims(sum(abs2.(field.E) .* grid.dt, dims=3); dims=3)
    copyto!(analyzer.Fxy, Fxygpu)

    Ftgpu = dropdims(sum(abs2.(field.E) .* grid.dx .* grid.dy, dims=(1,2)); dims=(1,2))
    copyto!(analyzer.Ft, Ftgpu)

    @views copyto!(analyzer.rho, field.rho[:,:,end])

    # Integral power spectrum:
    #     Sa = ifft(Ea)
    #     Sr = aspec2rspec(Sa)
    #     Sr = 2 * Sr * Nt * dt
    #     S = Int[|Sr|^2 * dx * dy]
    Nw = length(FFTW.rfftfreq(grid.Nt))
    field.PT \ field.E   # time -> frequency [exp(-i*w*t)]
    AnalyticSignals.aspec2rspec!(field.E)
    Sgpu = @views vec(
        sum(
            abs2.(2 * field.E[:,:,1:Nw] * grid.Nt * grid.dt) .* grid.dx .* grid.dy;
            dims=(1,2)
        )
    )
    copyto!(analyzer.S, Sgpu)
    AnalyticSignals.rspec2aspec!(field.E)
    field.PT * field.E   # frequency -> time [exp(-i*w*t)]


    analyzer.Fmax, imax = findmax(analyzer.Fxy)

    analyzer.Imax = maximum(abs2, field.E)

    analyzer.rhomax = maximum(field.rho)

    analyzer.De = sum(analyzer.rho .* grid.dx .* grid.dy)

    @views analyzer.ax = radius(grid.x, analyzer.Fxy[:, imax[2], 1])
    @views analyzer.ay = radius(grid.y, analyzer.Fxy[imax[1], :, 1])

    analyzer.W = sum(analyzer.Ft) * grid.dt
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
