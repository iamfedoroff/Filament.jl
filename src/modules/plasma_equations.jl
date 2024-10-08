function solve!(
    rho::AbstractArray{T,1},
    kdrho::AbstractArray{T,1},
    E::AbstractArray{Complex{T},1},
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
        utmp = ODEIntegrators.step(integ, utmp, t[j], dt, E)
        rho[j+1] = extract(utmp)

        kdrho_utmp = ODEIntegrators.step(kdrho_integ, kdrho_utmp, t[j], dt, E)
        kdrho[j] = kdrho_func(kdrho_utmp, kdrho_p, t[j], E)
    end
    return nothing
end


function solve!(
    rho::CUDA.CuArray{T,2},
    kdrho::CUDA.CuArray{T,2},
    E::CUDA.CuArray{Complex{T},2},
    t::AbstractArray{T,1},
    p::Tuple,
) where T<:AbstractFloat
    Nr, Nt = size(rho)

    ckernel = CUDA.@cuda launch=false solve_kernel_rt(rho, kdrho, E, t, p)
    config = CUDA.launch_configuration(ckernel.fun)
    threads = min(Nr, config.threads)
    blocks = cld(Nr, threads)

    ckernel(rho, kdrho, E, t, p; threads=threads, blocks=blocks)
    return nothing
end


function solve_kernel_rt(rho, kdrho, E, t, p)
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
            utmp = @views ODEIntegrators.step(integ, utmp, t[j], dt, E[i,:])
            rho[i, j+1] = extract(utmp)

            kdrho_utmp = @views ODEIntegrators.step(kdrho_integ, kdrho_utmp, t[j], dt, E[i,:])
            kdrho[i, j] = @views kdrho_func(kdrho_utmp, kdrho_p, t[j], E[i,:])
        end
    end
    return nothing
end


function solve!(
    rho::CUDA.CuArray{T,3},
    kdrho::CUDA.CuArray{T,3},
    E::CUDA.CuArray{Complex{T},3},
    t::AbstractArray{T,1},
    p::Tuple,
) where T<:AbstractFloat
    Nx, Ny, Nt = size(rho)

    ckernel = CUDA.@cuda launch=false solve_kernel_xyt(rho, kdrho, E, t, p)
    config = CUDA.launch_configuration(ckernel.fun)
    threads = min(Nx*Ny, config.threads)
    blocks = cld(Nx*Ny, threads)

    ckernel(rho, kdrho, E, t, p; threads=threads, blocks=blocks)
    return nothing
end


function solve_kernel_xyt(rho, kdrho, E, t, p)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x

    integs, extract, kdrho_integs, kdrho_func, kdrho_ps = p

    Nx, Ny, Nt = size(rho)
    dt = t[2] - t[1]

    cartesian = CartesianIndices((Nx, Ny, Nt))

    for I=id:stride:Nx*Ny
        ix = cartesian[I][1]
        iy = cartesian[I][2]

        integ = integs[ix,iy]
        kdrho_integ = kdrho_integs[ix,iy]
        kdrho_p = kdrho_ps[ix,iy]

        utmp = integ.prob.u0
        rho[ix,iy,1] = extract(utmp)
        kdrho_utmp = kdrho_integ.prob.u0
        for it=1:Nt-1
            utmp = @views ODEIntegrators.step(integ, utmp, t[it], dt, E[ix,iy,:])
            rho[ix,iy,it+1] = extract(utmp)

            kdrho_utmp = @views ODEIntegrators.step(kdrho_integ, kdrho_utmp, t[it], dt, E[ix,iy,:])
            kdrho[ix,iy,it] = @views kdrho_func(kdrho_utmp, kdrho_p, t[it], E[ix,iy,:])
        end
    end
    return nothing
end
