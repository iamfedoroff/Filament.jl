function solve!(
    rho::AbstractArray{T,1},
    kdrho::AbstractArray{T,1},
    t::AbstractArray{T,1},
    p::Tuple,
) where T<:AbstractFloat
    integ, extract, kdrho_integ, kdrho_func, kdrho_p = p

    Nt = length(t)
    dt = t[2] - t[1]

    utmp = integ.prob.u0
    rho[1] = extract(utmp)
    kdrho_utmp = kdrho_integ.prob.u0
    for j=1:Nt-1
        utmp = ODEIntegrators.step(integ, utmp, t[j], dt)
        rho[j+1] = extract(utmp)

        kdrho_utmp = ODEIntegrators.step(kdrho_integ, kdrho_utmp, t[j], dt)
        kdrho[j] = kdrho_func(kdrho_utmp, kdrho_p, t[j])
    end
    return nothing
end


function solve!(
    rho::CUDA.CuArray{T,2},
    kdrho::CUDA.CuArray{T,2},
    t::AbstractArray{T,1},
    p::Tuple,
) where T<:AbstractFloat
    Nr, Nt = size(rho)

    ckernel = CUDA.@cuda launch=false solve_kernel(rho, kdrho, t, p)
    config = CUDA.launch_configuration(ckernel.fun)
    threads = min(Nr, config.threads)
    blocks = cld(Nr, threads)

    ckernel(rho, kdrho, t, p; threads=threads, blocks=blocks)
    return nothing
end


function solve_kernel(rho, kdrho, t, p)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x

    integs, extract, kdrho_integs, kdrho_func, kdrho_ps = p

    Nr, Nt = size(rho)
    dt = t[2] - t[1]

    for i=id:stride:Nr
        integ = integs[i]
        kdrho_integ = kdrho_integs[i]
        kdrho_p = kdrho_ps[i]

        utmp = integ.prob.u0
        rho[i, 1] = extract(utmp)
        kdrho_utmp = kdrho_integ.prob.u0
        for j=1:Nt-1
            utmp = ODEIntegrators.step(integ, utmp, t[j], dt)
            rho[i, j+1] = extract(utmp)

            kdrho_utmp = ODEIntegrators.step(kdrho_integ, kdrho_utmp, t[j], dt)
            kdrho[i, j] = kdrho_func(kdrho_utmp, kdrho_p, t[j])
        end
    end
    return nothing
end
