module Fields

import AnalyticSignals
import CUDA
import FFTW
import HankelTransforms

import ..Constants: C0
import ..Grids
import ..Units


struct Field{
    T<:AbstractFloat,
    A<:AbstractArray{Complex{T}},
    TPS<:Union{HankelTransforms.Plan, FFTW.Plan, Nothing},
    TPT<:Union{FFTW.Plan, Nothing},
    AP<:Union{AbstractArray{T}, Nothing},
}
    w0 :: T
    E :: A
    PS :: TPS
    PT :: TPT
    rho :: AP
    kdrho :: AP
end



# ******************************************************************************
function Field(unit::Units.UnitR, grid::Grids.GridR, p::Tuple)
    lam0, initial_condition, HTLOAD, file_ht = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.r, unit.r, unit.I)
    E = CUDA.CuArray{Complex{T}}(E)

    if HTLOAD
        PS = HankelTransforms.plan(file_ht)
    else
        PS = HankelTransforms.plan(grid.rmax, E, save=true, fname="ht.jld2")
    end

    PT = nothing
    rho = nothing
    kdrho = nothing
    return Field(w0, E, PS, PT, rho, kdrho)
end


function Field(unit::Units.UnitT, grid::Grids.GridT, p::Tuple)
    lam0, initial_condition = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.t, unit.t, unit.I)
    E = Array{Complex{T}}(E)

    PT = FFTW.plan_fft!(E)

    AnalyticSignals.rsig2asig!(E, PT)   # convert to analytic signal

    PS = nothing
    rho = zeros(T, grid.Nt)
    kdrho = zeros(T, grid.Nt)

    # Initialize a dummy GPU array in order to trigger the creation of the
    # device context. This will allow to call CUDA.synchronize() in the
    # main cycle.
    tmp = CUDA.zeros(T, 1)

    return Field(w0, E, PS, PT, rho, kdrho)
end


function Field(unit::Units.UnitRT, grid::Grids.GridRT, p::Tuple)
    lam0, initial_condition, HTLOAD, file_ht = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    E = CUDA.CuArray{Complex{T}}(E)

    if HTLOAD
        PS = HankelTransforms.plan(file_ht)
    else
        Nthalf = AnalyticSignals.half(grid.Nt)
        region = CartesianIndices((grid.Nr, Nthalf))
        PS = HankelTransforms.plan(
            grid.rmax, E, region, save=true, fname="ht.jld2",
        )
    end

    # in-place FFTs results in segfault after run completion
    # https://github.com/JuliaGPU/CUDA.jl/issues/95
    # PT = FFTW.plan_fft!(E, [2])
    PT = FFTW.plan_fft(E, [2])

    AnalyticSignals.rsig2asig!(E, PT)   # convert to analytic signal

    rho = CUDA.zeros(T, (grid.Nr, grid.Nt))
    kdrho = CUDA.zeros(T, (grid.Nr, grid.Nt))
    return Field(w0, E, PS, PT, rho, kdrho)
end


function Field(unit::Units.UnitXY, grid::Grids.GridXY, p::Tuple)
    lam0, initial_condition = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.x, grid.y, unit.x, unit.y, unit.I)
    E = CUDA.CuArray{Complex{T}}(E)

    # in-place FFTs results in segfault after run completion
    # https://github.com/JuliaGPU/CUDA.jl/issues/95
    # PS = FFTW.plan_fft!(E)
    PS = FFTW.plan_fft(E)

    PT = nothing
    rho = nothing
    kdrho = nothing
    return Field(w0, E, PS, PT, rho, kdrho)
end


function Field(unit::Units.UnitXYT, grid::Grids.GridXYT, p::Tuple)
    lam0, initial_condition = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(
        grid.x, grid.y, grid.t, unit.x, unit.y, unit.t, unit.I,
    )
    E = CUDA.CuArray{Complex{T}}(E)

    # in-place FFTs results in segfault after run completion
    # https://github.com/JuliaGPU/CUDA.jl/issues/95
    # PS = FFTW.plan_fft!(E, [1, 2])
    # PT = FFTW.plan_fft!(E, [3])
    PS = FFTW.plan_fft(E, [1, 2])
    PT = FFTW.plan_fft(E, [3])

    AnalyticSignals.rsig2asig!(E, PT)   # convert to analytic signal

    rho = CUDA.zeros(T, (grid.Nx, grid.Ny, grid.Nt))
    kdrho = CUDA.zeros(T, (grid.Nx, grid.Ny, grid.Nt))
    return Field(w0, E, PS, PT, rho, kdrho)
end


end
