module Equations

import StaticArrays


struct PFunction{F<:Function, T<:Tuple} <: Function
    func :: F
    p :: T
end


function (pfunc::PFunction)(x...)
    pfunc.func(x..., pfunc.p)
end


struct Term{T<:AbstractFloat, F<:Function, I<:Int}
    R :: T
    calculate :: F
    Neq :: I
end


struct Problem{T<:AbstractArray, F<:Function}
    u0 :: T
    step :: F
end


function Problem(alg::String, u0::AbstractArray, func::Function)
    @assert alg in ("RK2", "RK3", "RK4")

    if typeof(u0) <: StaticArrays.SVector
        if alg == "RK2"
            step = rk2
        elseif alg == "RK3"
            step = rk3
        elseif alg == "RK4"
            step = rk4
        end
        p = (func, )
    else
        tmp = similar(u0)
        if alg == "RK2"
            step = rk2!
            k1 = similar(u0)
            k2 = similar(u0)
            p = (func, k1, k2, tmp)
        elseif alg == "RK3"
            step = rk3!
            k1 = similar(u0)
            k2 = similar(u0)
            k3 = similar(u0)
            p = (func, k1, k2, k3, tmp)
        elseif alg == "RK4"
            step = rk4!
            k1 = similar(u0)
            k2 = similar(u0)
            k3 = similar(u0)
            k4 = similar(u0)
            p = (func, k1, k2, k3, k4, tmp)
        end
    end

    pstep = PFunction(step, p)
    return Problem(u0, pstep)
end


function rk2(u::StaticArrays.SVector, dt::AbstractFloat, args::Tuple, p::Tuple)
    func, = p

    k1 = func(u, args)

    ktmp = u + dt * 2 / 3 * k1
    k2 = func(ktmp, args)

    return u + dt / 4 * (k1 + 3 * k2)
end


function rk2!(u::AbstractArray, dt::AbstractFloat, args::Tuple, p::Tuple)
    func, k1, k2, tmp = p

    func(k1, u, args)

    @. tmp = u + dt * 2 / 3 * k1
    func(k2, tmp, args)

    @. u = u + dt / 4 * (k1 + 3 * k2)
    return nothing
end


function rk3(u::StaticArrays.SVector, dt::AbstractFloat, args::Tuple, p::Tuple)
    func, = p

    k1 = func(u, args)

    ktmp = u + dt / 2 * k1
    k2 = func(ktmp, args)

    ktmp = u + dt * (-1 * k1 + 2 * k2)
    k3 = func(ktmp, args)

    return u + dt / 6 * (k1 + 4 * k2 + k3)
end


function rk3!(u::AbstractArray, dt::AbstractFloat, args::Tuple, p::Tuple)
    func, k1, k2, k3, tmp = p

    func(k1, u, args)

    @. tmp = u + dt / 2 * k1
    func(k2, tmp, args)

    @. tmp = u + dt * (-1 * k1 + 2 * k2)
    func(k3, tmp, args)

    @. u = u + dt / 6 * (k1 + 4 * k2 + k3)
    return nothing
end


function rk4(u::StaticArrays.SVector, dt::AbstractFloat, args::Tuple, p::Tuple)
    func, = p

    k1 = func(u, args)

    ktmp = u + dt / 2 * k1
    k2 = func(ktmp, args)

    ktmp = u + dt / 2 * k2
    k3 = func(ktmp, args)

    ktmp = u + dt * k3
    k4 = func(ktmp, args)

    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
end


function rk4!(u::AbstractArray, dt::AbstractFloat, args::Tuple, p::Tuple)
    func, k1, k2, k3, k4, tmp = p

    func(k1, u, args)

    @. tmp = u + dt / 2 * k1
    func(k2, tmp, args)

    @. tmp = u + dt / 2 * k2
    func(k3, tmp, args)

    @. tmp = u + dt * k3
    func(k4, tmp, args)

    @. u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return nothing
end


end
