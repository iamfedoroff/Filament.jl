module PlasmaEquations

import CUDAnative
import CuArrays
import StaticArrays

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


function solve!(
    rho::AbstractArray{T,1},
    kdrho::AbstractArray{T,1},
    t::AbstractArray{T,1},
    E::AbstractArray{Complex{T},1},
    p::Tuple,
) where T<:AbstractFloat
    prob, extract, kdrho_func = p
    Nt = length(rho)
    dt = t[2] - t[1]
    utmp = prob.u0
    rho[1] = extract(prob.u0)
    for j=1:Nt-1
        args = (E[j], )
        utmp, tnew = prob.step(utmp, t[j], dt, args)
        while tnew < t[j] + dt   # not t[j+1] to avoid truncation errors
            dtnew = t[j+1] - tnew
            utmp, tnew = prob.step(utmp, tnew, dtnew, args)
        end
        rho[j+1] = extract(utmp)
        kdrho[j] = kdrho_func(utmp, t[j], args)
    end
    return nothing
end


function solve!(
    rho::AbstractArray{T,2},
    kdrho::AbstractArray{T,2},
    t::AbstractArray{T,1},
    E::AbstractArray{Complex{T},2},
    p::Tuple,
) where T<:AbstractFloat
    prob, extract, kdrho_func = p
    Nr, Nt = size(rho)
    dt = t[2] - t[1]
    for i=1:Nr
        utmp = prob.u0
        rho[i, 1] = extract(prob.u0)
        for j=1:Nt-1
            args = (E[i, j], )
            utmp, tnew = prob.step(utmp, t[j], dt, args)
            while tnew < t[j] + dt   # not t[j+1] to avoid truncation errors
                dtnew = t[j+1] - tnew
                utmp, tnew = prob.step(utmp, tnew, dtnew, args)
            end
            rho[i, j+1] = extract(utmp)
            kdrho[i, j] = kdrho_func(utmp, t[j], args)
        end
    end
    return nothing
end


function solve!(
    rho::CuArrays.CuArray{T,2},
    kdrho::CuArrays.CuArray{T,2},
    t::AbstractArray{T,1},
    E::CuArrays.CuArray{Complex{T},2},
    p::Tuple,
) where T<:AbstractFloat
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
        utmp = prob.u0
        rho[i, 1] = extract(prob.u0)
        for j=1:Nt-1
            args = (E[i, j], )
            utmp, tnew = prob.step(utmp, t[j], dt, args)
            while tnew < t[j] + dt   # not t[j+1] to avoid truncation errors
                dtnew = t[j+1] - tnew
                utmp, tnew = prob.step(utmp, tnew, dtnew, args)
            end
            rho[i, j+1] = extract(utmp)
            kdrho[i, j] = kdrho_func(utmp, t[j], args)
        end
    end
    return nothing
end

end
