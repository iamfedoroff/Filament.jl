module RungeKuttas

import CuArrays


abstract type RungeKutta end


struct RungeKutta2{T} <: RungeKutta
    k1 :: CuArrays.CuArray{T}
    k2 :: CuArrays.CuArray{T}
    tmp :: CuArrays.CuArray{T}
end


struct RungeKutta3{T} <: RungeKutta
    k1 :: CuArrays.CuArray{T}
    k2 :: CuArrays.CuArray{T}
    k3 :: CuArrays.CuArray{T}
    tmp :: CuArrays.CuArray{T}
end


struct RungeKutta4{T} <: RungeKutta
    k1 :: CuArrays.CuArray{T}
    k2 :: CuArrays.CuArray{T}
    k3 :: CuArrays.CuArray{T}
    k4 :: CuArrays.CuArray{T}
    tmp :: CuArrays.CuArray{T}
end


function RungeKutta(order::Integer, T::Type, ndims::Integer...)
    if order == 2
        k1 = CuArrays.cuzeros(T, ndims)
        k2 = CuArrays.cuzeros(T, ndims)
        tmp = CuArrays.cuzeros(T, ndims)
        RK = RungeKutta2(k1, k2, tmp)
    elseif order == 3
        k1 = CuArrays.cuzeros(T, ndims)
        k2 = CuArrays.cuzeros(T, ndims)
        k3 = CuArrays.cuzeros(T, ndims)
        tmp = CuArrays.cuzeros(T, ndims)
        RK = RungeKutta3(k1, k2, k3, tmp)
    elseif order == 4
        k1 = CuArrays.cuzeros(T, ndims)
        k2 = CuArrays.cuzeros(T, ndims)
        k3 = CuArrays.cuzeros(T, ndims)
        k4 = CuArrays.cuzeros(T, ndims)
        tmp = CuArrays.cuzeros(T, ndims)
        RK = RungeKutta4(k1, k2, k3, k4, tmp)
    else
        println("ERROR: Wrong Runge-Kutta order.")
        exit()
    end
    return RK
end


function solve!(RK::RungeKutta2, u::CuArrays.CuArray{Complex{T}}, h::T,
                func!::Function, p::Tuple) where T
    func!(RK.k1, u, p)

    @. RK.tmp = u + h * 2. / 3. * RK.k1
    func!(RK.k2, RK.tmp, p)

    @. u = u + h / 4. * (RK.k1 + 3. * RK.k2)
    return nothing
end


function solve!(RK::RungeKutta3, u::CuArrays.CuArray{Complex{T}}, h::T,
                func!::Function, p::Tuple) where T
    func!(RK.k1, u, p)

    @. RK.tmp = u + h * 0.5 * RK.k1
    func!(RK.k2, RK.tmp, p)

    @. RK.tmp = u + h * (-1. * RK.k1 + 2. * RK.k2)
    func!(RK.k3, RK.tmp, p)

    @. u = u + h / 6. * (RK.k1 + 4. * RK.k2 + RK.k3)
    return nothing
end


function solve!(RK::RungeKutta4, u::CuArrays.CuArray{Complex{T}}, h::T,
                func!::Function, p::Tuple) where T
    func!(RK.k1, u, p)

    @. RK.tmp = u + h * 0.5 * RK.k1
    func!(RK.k2, RK.tmp, p)

    @. RK.tmp = u + h * 0.5 * RK.k2
    func!(RK.k3, RK.tmp, p)

    @. RK.tmp = u + h * RK.k3
    func!(RK.k4, RK.tmp, p)

    @. u = u + h / 6. * (RK.k1 + 2. * RK.k2 + 2. * RK.k3 + RK.k4)
    return nothing
end


end
