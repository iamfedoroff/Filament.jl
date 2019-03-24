# ******************************************************************************
# Losses due to multiphoton ionization
# ******************************************************************************
function init_current_losses(unit, grid, field, medium, plasma, args)
    IONARG = args["IONARG"]

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

    p = (IONARG, plasma.Kdrho)

    return Rnl, calculate_current_losses, p
end


function calculate_current_losses(F::CuArrays.CuArray{FloatGPU, 2},
                                  E::CuArrays.CuArray{ComplexGPU, 2},
                                  p::Tuple)
    IONARG = p[1]
    Kdrho = p[2]
    if IONARG != 0
        @. F = abs2(E)
    else
        @. F = real(E)^2
    end
    inverse!(F)
    @. F = Kdrho * F * real(E)
end


function inverse!(F::CuArrays.CuArray{FloatGPU, 2})
    N1, N2 = size(F)
    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nth = min(N1 * N2, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N1 * N2 / nth))
    @CUDAnative.cuda blocks=nbl threads=nth inverse_kernel(F)
    return nothing
end


function inverse_kernel(F)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(F)
    for k=id:stride:N1*N2
        if F[k] >= FloatGPU(1e-30)
            F[k] = FloatGPU(1.) / F[k]
        else
            F[k] = FloatGPU(0.)
        end
    end
    return nothing
end
