# ******************************************************************************
# Losses due to multiphoton ionization
# ******************************************************************************
function init_current_losses(unit, grid, field, medium, p)
    EREAL = p["EREAL"]

    n0 = Media.refractive_index(medium, field.w0)
    Eu = Units.E(unit, real(n0))

    Rnl = zeros(ComplexF64, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            Rnl[i] = 1im / (grid.w[i] * unit.w) *
                     HBAR * field.w0 * unit.rho / (unit.t * Eu)
        end
    end

    @. Rnl = conj(Rnl)
    Rnl = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, Rnl))

    fearg_real(x::Complex{FloatGPU}) = real(x)^2
    fearg_abs2(x::Complex{FloatGPU}) = abs2(x)
    if EREAL
        fearg = fearg_real
    else
        fearg = fearg_abs2
    end

    p_calc = (field.Kdrho, fearg)
    pcalc = Equations.PFunction(calc_current_losses, p_calc)

    p_dzadapt = ()
    pdzadapt = Equations.PFunction(dzadapt_current_losses, p_dzadapt)

    return Media.NonlinearResponse(Rnl, pcalc, pdzadapt)
end


function calc_current_losses(F::CuArrays.CuArray{T},
                             E::CuArrays.CuArray{Complex{T}},
                             args::Tuple,
                             p::Tuple) where T
    Kdrho, fearg = p
    @. F = fearg(E)
    inverse!(F)
    @. F = Kdrho * F * real(E)
    return nothing
end


function inverse!(F::CuArrays.CuArray{T}) where T
    N = length(F)
    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
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


function dzadapt_current_losses(phimax::AbstractFloat, p::Tuple)
    return Inf
end
