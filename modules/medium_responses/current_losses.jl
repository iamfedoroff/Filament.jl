# ******************************************************************************
# Losses due to multiphoton ionization
# ******************************************************************************
function init_current_losses(unit, grid, field, medium, args)
    EREAL = args["EREAL"]

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

    p = (field.Kdrho, )

    if EREAL
        calc = calc_current_losses_real
    else
        calc = calc_current_losses_abs2
    end

    return Rnl, calc, p
end


function calc_current_losses_abs2(z::T,
                                  F::CuArrays.CuArray{T},
                                  E::CuArrays.CuArray{Complex{T}},
                                  p::Tuple) where T
    Kdrho = p[1]
    @. F = abs2(E)
    inverse!(F)
    @. F = Kdrho * F * real(E)
end


function calc_current_losses_real(z::T,
                                  F::CuArrays.CuArray{T},
                                  E::CuArrays.CuArray{Complex{T}},
                                  p::Tuple) where T
    Kdrho = p[1]
    @. F = real(E)^2
    inverse!(F)
    @. F = Kdrho * F * real(E)
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
