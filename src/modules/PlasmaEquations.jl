module PlasmaEquations

import CUDAnative
import CuArrays
import StaticArrays

import Equations
import Units


struct PlasmaEquation{I, FE, FK, P}
    integ :: I
    extract :: FE
    func_kdrho :: FK
    p_kdrho :: P
end


function PlasmaEquation(unit::Units.Unit, n0, w0, params)
    init = params["init"]
    integ, extract, func_kdrho, p_kdrho = init(unit, n0, w0, params)
    return PlasmaEquation(integ, extract, func_kdrho, p_kdrho)
end


function solve!(
    PE::PlasmaEquation,
    rho::AbstractArray{T,1},
    kdrho::AbstractArray{T,1},
    t::AbstractArray{T,1},
    E::AbstractArray{Complex{T},1},
) where T<:AbstractFloat
    integ = PE.integ
    extract = PE.extract
    func_kdrho = PE.func_kdrho
    p_kdrho = PE.p_kdrho

    Nt = length(t)
    dt = t[2] - t[1]

    utmp = integ.prob.u0
    rho[1] = extract(utmp)
    for j=1:Nt-1
        args = (E[j], )
        utmp = Equations.step(integ, utmp, t[j], dt, args)
        rho[j+1] = extract(utmp)
        kdrho[j] = func_kdrho(utmp, p_kdrho, t[j], args)
    end
    return nothing
end


function solve!(
    PE::PlasmaEquation,
    rho::AbstractArray{T,2},
    kdrho::AbstractArray{T,2},
    t::AbstractArray{T,1},
    E::AbstractArray{Complex{T},2},
) where T<:AbstractFloat
    integ = PE.integ
    extract = PE.extract
    func_kdrho = PE.func_kdrho
    p_kdrho = PE.p_kdrho

    dt = t[2] - t[1]

    Nr, Nt = size(rho)
    for i=1:Nr
        utmp = integ.prob.u0
        rho[i, 1] = extract(utmp)
        for j=1:Nt-1
            args = (E[i, j], )
            utmp = Equations.step(integ, utmp, t[j], dt, args)
            rho[i, j+1] = extract(utmp)
            kdrho[i, j] = func_kdrho(utmp, p_kdrho, t[j], args)
        end
    end
    return nothing
end


function solve!(
    PE::PlasmaEquation,
    rho::CuArrays.CuArray{T,2},
    kdrho::CuArrays.CuArray{T,2},
    t::AbstractArray{T,1},
    E::CuArrays.CuArray{Complex{T},2},
) where T<:AbstractFloat
    Nr, Nt = size(rho)
    nth = min(256, Nr)
    nbl = Int(ceil(Nr / nth))
    @CUDAnative.cuda blocks=nbl threads=nth solve_kernel(PE, rho, kdrho, t, E)
    return nothing
end

function solve_kernel(PE, rho, kdrho, t, E)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x

    integ = PE.integ
    extract = PE.extract
    func_kdrho = PE.func_kdrho
    p_kdrho = PE.p_kdrho

    dt = t[2] - t[1]

    Nr, Nt = size(rho)
    for i=id:stride:Nr
        utmp = integ.prob.u0
        rho[i, 1] = extract(utmp)
        for j=1:Nt-1
            args = (E[i, j], )
            utmp = Equations.step(integ, utmp, t[j], dt, args)
            rho[i, j+1] = extract(utmp)
            kdrho[i, j] = func_kdrho(utmp, p_kdrho, t[j], args)
        end
    end
    return nothing
end


end
