# ******************************************************************************
# Losses due to multiphoton ionization
# ******************************************************************************
function init_current_losses(unit, grid, field, medium, p)
    EREAL = p["EREAL"]

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    Rnl = zeros(ComplexF64, grid.Nt)
    for i=1:grid.Nt
        if grid.w[i] != 0
            Rnl[i] = 1im / (grid.w[i] * unit.w) *
                     HBAR * w0 * unit.rho / (unit.t * Eu)
        end
    end

    @. Rnl = conj(Rnl)
    if !(typeof(grid) <: Grids.GridT)   # FIXME: should be removed in a generic code.
        Rnl = CuArrays.CuArray{Complex{FloatGPU}}(Rnl)
    end

    fearg_real(x::Complex) = real(x)^2
    fearg_abs2(x::Complex) = abs2(x)
    if EREAL
        fearg = fearg_real
    else
        fearg = fearg_abs2
    end

    p = (field.kdrho, fearg)
    return Media.NonlinearResponse(Rnl, calc_current_losses, p)
end


function calc_current_losses(
    F::AbstractArray{Complex{T}}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    kdrho, fearg = p
    @. F = fearg(E)
    inverse!(F)
    @. F = kdrho * F * real(E)
    return nothing
end


function inverse!(F::AbstractArray{Complex{T}}) where T<:AbstractFloat
    for i=1:length(F)
        if real(F[i]) >= 1e-30
            F[i] = 1 / F[i]
        else
            F[i] = 0
        end
    end
end


function inverse!(F::CuArrays.CuArray{Complex{T}}) where T<:AbstractFloat
    N = length(F)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth inverse_kernel(F)
    return nothing
end


function inverse_kernel(F)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(F)
    for k=id:stride:N
        if real(F[k]) >= 1e-30
            F[k] = 1 / F[k]
        else
            F[k] = 0
        end
    end
    return nothing
end
