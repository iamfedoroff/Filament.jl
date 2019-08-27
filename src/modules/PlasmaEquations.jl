module PlasmaEquations

import StaticArrays
import CuArrays
import CUDAnative

import Equations


struct PlasmaEquation{T <: Function}
    solve! :: T
end


function PlasmaEquation(unit, n0, w0, params)
    init = params["init"]
    prob, extract, kdrho_func = init(unit, n0, w0, params)

    p = (prob, extract, kdrho_func)
    psolve! = Equations.PFunction(solve!, p)
    return PlasmaEquation(psolve!)
end


function solve!(rho::AbstractArray{T,1},
                kdrho::AbstractArray{T,1},
                t::AbstractArray{T,1},
                E::AbstractArray{Complex{T},1},
                p::Tuple) where T<:AbstractFloat
    prob, extract, kdrho_func = p
    Nt = length(rho)
    dt = t[2] - t[1]
    tmp = prob.u0
    rho[1] = extract(prob.u0)
    for j=1:Nt-1
        args = (E[j], )
        tmp = prob.step(tmp, t[j], dt, args)
        rho[j+1] = extract(tmp)
        kdrho[j] = kdrho_func(tmp, t[j], args)
    end
    return nothing
end


function solve!(rho::AbstractArray{T,2},
                kdrho::AbstractArray{T,2},
                t::AbstractArray{T,1},
                E::AbstractArray{Complex{T},2},
                p::Tuple) where T<:AbstractFloat
    prob, extract, kdrho_func = p
    Nr, Nt = size(rho)
    dt = t[2] - t[1]
    for i=1:Nr
        tmp = prob.u0
        rho[i, 1] = extract(prob.u0)
        for j=1:Nt-1
            args = (E[i, j], )
            tmp = prob.step(tmp, t[j], dt, args)
            rho[i, j+1] = extract(tmp)
            kdrho[i, j] = kdrho_func(tmp, t[j], args)
        end
    end
    return nothing
end


function solve!(rho::CuArrays.CuArray{T,2},
                kdrho::CuArrays.CuArray{T,2},
                t::AbstractArray{T,1},
                E::CuArrays.CuArray{Complex{T},2},
                p::Tuple) where T<:AbstractFloat
    Nr, Nt = size(rho)
    nth = min(256, Nr)
    nbl = Int(ceil(Nr / nth))
    @CUDAnative.cuda blocks=nbl threads=nth solve_kernel(rho, kdrho, t, E, p)
    return nothing
end


function solve_kernel(rho, kdrho, t, E, p)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    prob, extract, kdrho_func = p
    Nr, Nt = size(rho)
    dt = t[2] - t[1]
    for i=id:stride:Nr
        tmp = prob.u0
        rho[i, 1] = extract(prob.u0)
        for j=1:Nt-1
            args = (E[i, j], )
            tmp = prob.step(tmp, t[j], dt, args)
            rho[i, j+1] = extract(tmp)
            kdrho[i, j] = kdrho_func(tmp, t[j], args)
        end
    end
    return nothing
end

end
