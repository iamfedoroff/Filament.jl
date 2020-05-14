module Models

# Global packages:
import CuArrays
import CUDAdrv
import CUDAnative
import HankelTransforms
import StaticArrays
using TimerOutputs

# Local package-like modules:
import ..AnalyticSignals
import ..FourierTransforms
import ..Equations

# Local modules:
import ..Constants: FloatGPU, MU0
import ..Fields
import ..Grids
import ..Guards
import ..Media
import ..Units

include("linear_propagators.jl")
include("nonlinear_propagators.jl")
include("plasma_equations.jl")


struct Model{TLP, TNP, TPE, B<:Bool}
    LP :: TLP
    NP :: TNP
    PE :: TPE
    NONLINEARITY :: B
    PLASMA :: B
end


function Model(
    unit::Units.Unit,
    grid::Grids.Grid,
    field::Fields.Field,
    medium::Media.Medium,
    guard::Guards.Guard,
    responses::AbstractArray,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    NONLINEARITY = keys.NONLINEARITY
    PLASMA = keys.PLASMA

    LP = LinearPropagator(unit, grid, medium, field, guard, keys.KPARAXIAL)

    if NONLINEARITY
        NP = NonlinearPropagator(
            unit, grid, medium, field, guard, responses, keys.QPARAXIAL,
            keys.ALG,
        )
    else
        NP = nothing
    end

    if PLASMA
        w0 = field.w0
        n0 = Media.refractive_index(medium, w0)
        PE = PlasmaEquation(unit, n0, w0, plasma_equation)
        solve!(PE, field.rho, field.kdrho, grid.t, field.E)
    else
        PE = nothing
    end

    return Model(LP, NP, PE, NONLINEARITY, PLASMA)
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.Grid,
    field::Fields.Field,
    guard::Guards.Guard,
    model::Model,
) where T<:AbstractFloat
    if model.PLASMA
        @timeit "plasma" begin
            solve!(model.PE, field.rho, field.kdrho, grid.t, field.E)
            CUDAdrv.synchronize()
        end
    end

    if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT)
        @timeit "field -> spectr" begin
            forward_transform_time!(field.E, field.PT)
            CUDAdrv.synchronize()
        end
    end

    if model.NONLINEARITY
        @timeit "nonlinearity" begin
           propagate!(field.E, model.NP, z, dz)
           CUDAdrv.synchronize()
       end
    end

    @timeit "linear" begin
        propagate!(field.E, model.LP, dz)
        CUDAdrv.synchronize()
    end

    if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT)
        @timeit "spectr -> field" begin
            inverse_transform_time!(field.E, field.PT)
            CUDAdrv.synchronize()
        end
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(field.E, guard)
        CUDAdrv.synchronize()
    end
    return nothing
end


# ******************************************************************************
function forward_transform_space!(E::AbstractArray, P::Nothing)
    return nothing
end


function inverse_transform_space!(E::AbstractArray, P::Nothing)
    return nothing
end


function forward_transform_space!(E::AbstractArray, P::HankelTransforms.Plan)
    HankelTransforms.dht!(E, P)
    return nothing
end


function inverse_transform_space!(E::AbstractArray, P::HankelTransforms.Plan)
    HankelTransforms.idht!(E, P)
    return nothing
end


function forward_transform_space!(E::AbstractArray, P::FourierTransforms.Plan)
    FourierTransforms.fft!(E, P)
    return nothing
end


function inverse_transform_space!(E::AbstractArray, P::FourierTransforms.Plan)
    FourierTransforms.ifft!(E, P)
    return nothing
end


# ******************************************************************************
function forward_transform_time!(E::AbstractArray, P::Nothing)
    return nothing
end


function inverse_transform_time!(E::AbstractArray, P::Nothing)
    return nothing
end


function forward_transform_time!(E::AbstractArray, P::FourierTransforms.Plan)
    FourierTransforms.fft!(E, P)
    return nothing
end


function inverse_transform_time!(E::AbstractArray, P::FourierTransforms.Plan)
    FourierTransforms.ifft!(E, P)
    return nothing
end


# ******************************************************************************
function real_signal_to_analytic_spectrum!(E::AbstractArray, P::Nothing)
    return nothing
end


function real_signal_to_analytic_spectrum!(
    E::AbstractArray, P::FourierTransforms.Plan,
)
    AnalyticSignals.rsig2aspec!(E, P)
    return nothing
end


end
