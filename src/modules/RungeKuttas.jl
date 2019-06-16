module RungeKuttas


function Problem(alg::String, u0::Union{AbstractFloat,AbstractArray}, func::Function, p::Tuple)
    @assert alg in ("RK2", "RK3", "RK4")
    if typeof(u0) <: AbstractFloat
        if alg == "RK2"
            stepfunc = rk2
        elseif alg == "RK3"
            stepfunc = rk3
        elseif alg == "RK4"
            stepfunc = rk4
        end
        cache = ()
    else
        tmp = similar(u0)
        if alg == "RK2"
            stepfunc = rk2!
            k1 = similar(u0)
            k2 = similar(u0)
            cache = (k1, k2, tmp)
        elseif alg == "RK3"
            stepfunc = rk3!
            k1 = similar(u0)
            k2 = similar(u0)
            k3 = similar(u0)
            cache = (k1, k2, k3, tmp)
        elseif alg == "RK4"
            stepfunc = rk4!
            k1 = similar(u0)
            k2 = similar(u0)
            k3 = similar(u0)
            k4 = similar(u0)
            cache = (k1, k2, k3, k4, tmp)
        end
    end
    return (u0=u0, stepfunc=stepfunc, cache=cache, func=func, p=p)
end


function step(prob::NamedTuple, u, dt::AbstractFloat, args::Tuple=())
    prob.stepfunc(u, dt, prob.cache, prob.func, prob.p, args)
end


function rk2(u::T, dt::T, cache::Tuple, func::Function, p::Tuple, args::Tuple) where T<:AbstractFloat
    k1 = func(u, p, args)
    k2 = func(u + dt * 2 / 3 * k1, p, args)
    return u + dt / 4 * (k1 + 3 * k2)
end


function rk2!(u::AbstractArray, dt::AbstractFloat, cache::Tuple, func::Function, p::Tuple, args::Tuple)
    k1, k2, tmp = cache

    func(k1, u, p, args)

    @. tmp = u + dt * 2 / 3 * k1
    func(k2, tmp, p, args)

    @. u = u + dt / 4 * (k1 + 3 * k2)
    return nothing
end


function rk3(u::T, dt::T, cache::Tuple, func::Function, p::Tuple, args::Tuple) where T<:AbstractFloat
    k1 = func(u, p, args)
    k2 = func(u + dt / 2 * k1, p, args)
    k3 = func(u + dt * (-1 * k1 + 2 * k2), p, args)
    return u + dt / 6 * (k1 + 4 * k2 + k3)
end


function rk3!(u::AbstractArray, dt::AbstractFloat, cache::Tuple, func::Function, p::Tuple, args::Tuple)
    k1, k2, k3, tmp = cache

    func(k1, u, p, args)

    @. tmp = u + dt / 2 * k1
    func(k2, tmp, p, args)

    @. tmp = u + dt * (-1 * k1 + 2 * k2)
    func(k3, tmp, p, args)

    @. u = u + dt / 6 * (k1 + 4 * k2 + k3)
    return nothing
end


function rk4(u::T, dt::T, cache::Tuple, func::Function, p::Tuple, args::Tuple) where T<:AbstractFloat
    k1 = func(u, p, args)
    k2 = func(u + dt / 2 * k1, p, args)
    k3 = func(u + dt / 2 * k2, p, args)
    k4 = func(u + dt * k3, p, args)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
end


function rk4!(u::AbstractArray, dt::AbstractFloat, cache::Tuple, func::Function, p::Tuple, args::Tuple)
    k1, k2, k3, k4, tmp = cache

    func(k1, u, p, args)

    @. tmp = u + dt / 2 * k1
    func(k2, tmp, p, args)

    @. tmp = u + dt / 2 * k2
    func(k3, tmp, p, args)

    @. tmp = u + dt * k3
    func(k4, tmp, p, args)

    @. u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return nothing
end


end
