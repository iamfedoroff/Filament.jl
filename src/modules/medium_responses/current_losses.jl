# ******************************************************************************
# Losses due to multiphoton ionization
# ******************************************************************************
function init_current_losses(unit, grid, field, medium, p)
    EREAL = p["EREAL"]

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)
    Eu = Units.E(unit, real(n0))

    Rnl = zeros(ComplexF64, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            Rnl[i] = 1im / (grid.w[i] * unit.w) *
                     HBAR * w0 * unit.rho / (unit.t * Eu)
        end
    end

    @. Rnl = conj(Rnl)
    if grid.geometry != "T"   # FIXME: should be removed in a generic code.
        Rnl = CuArrays.CuArray(convert(Array{Complex{FloatGPU}, 1}, Rnl))
    end

    fearg_real(x::Complex) = real(x)^2
    fearg_abs2(x::Complex) = abs2(x)
    if EREAL
        fearg = fearg_real
    else
        fearg = fearg_abs2
    end

    p = (field.Kdrho, fearg)
    return Media.NonlinearResponse(Rnl, calc_current_losses, p)
end


function calc_current_losses(
    F::AbstractArray{T}, E::AbstractArray{Complex{T}}, p::Tuple, z::T,
) where T<:AbstractFloat
    Kdrho, fearg = p
    @. F = fearg(E)
    inverse!(F)
    @. F = Kdrho * F * real(E)
    return nothing
end


function inverse!(F::AbstractArray{T}) where T<:AbstractFloat
    for i=1:length(F)
        if F[i] >= 1e-30
            F[i] = 1 / F[i]
        else
            F[i] = 0
        end
    end
end


function inverse!(F::CuArrays.CuArray{T}) where T<:AbstractFloat
    N = length(F)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
    @CUDAnative.cuda blocks=nbl threads=nth inverse_kernel(F)
    return nothing
end


function inverse_kernel(F)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(F)
    for k=id:stride:N
        if F[k] >= 1e-30
            F[k] = 1 / F[k]
        else
            F[k] = 0
        end
    end
    return nothing
end
