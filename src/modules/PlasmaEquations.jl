module PlasmaEquations

using Unrolled
import StaticArrays
import CUDAnative
import CuArrays

import Units
import Equations

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const QE = scipy_constants.e   # elementary charge [C]
const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32


struct PlasmaEquation
    solve! :: Function
end


function PlasmaEquation(unit::Units.Unit, dt, n0, w0, params::Dict)
    alg = params["ALG"]
    rho0 = params["rho0"]
    terms_list = params["terms"]
    kdrho_term_dict = params["Kdrho_term"]

    # Equation terms:
    Neq = 0
    terms = []
    for (i, item) in enumerate(terms_list)
        init = item["init"]
        term = init(unit, n0, w0, item)

        if i == 1
            Neq = term.Neq
        else
            @assert term.Neq == Neq
        end

        push!(terms, term)
    end
    terms = tuple(terms...)

    rho0 = ones(Neq) * rho0 / unit.rho
    rho0 = StaticArrays.SVector{Neq, FloatGPU}(rho0)
    szeros = StaticArrays.SVector{Neq, FloatGPU}(zeros(Neq))
    p = (terms, szeros)
    pstepfunc = Equations.PFunction(stepfunc, p)
    prob = Equations.Problem(alg, rho0, pstepfunc)

    # K * drho term:
    kdrho_init = kdrho_term_dict["init"]
    kdrho_term = kdrho_init(unit, n0, w0, kdrho_term_dict)

    components = kdrho_term_dict["components"]
    Neq = length(components)
    Ks = zeros(Neq)
    for (i, comp) in enumerate(components)
        Ui = comp["ionization_energy"]
        Ui = Ui * QE   # eV -> J
        Wph = HBAR * w0
        Ks[i] = ceil(Ui / Wph)   # order of multiphoton ionization
    end
    Ks = StaticArrays.SVector{Neq, FloatGPU}(Ks)

    # Problem:
    p = (prob, dt, kdrho_term, Ks)
    psolve! = Equations.PFunction(solve!, p)

    return PlasmaEquation(psolve!)
end


function stepfunc(rho::StaticArrays.SVector, args::Tuple, p::Tuple)
    terms, szeros = p
    return do_sum(terms, szeros, rho, args)
end


"""
The necessity of @unroll macro is explained here:
https://discourse.julialang.org/t/non-allocating-loop-over-a-set-of-structs/25643
"""
@unroll function do_sum(terms, szeros, rho, args)
    drho = szeros
    @unroll for term in terms
        drho = drho + term.R * term.calculate(rho, args)
    end
    return drho
end


function solve!(rho::AbstractArray{T},
                kdrho::AbstractArray{T},
                E::AbstractArray{Complex{T}},
                p::Tuple) where T<:AbstractFloat
    prob, dt, kdrho_term, Ks = p

    Nr, Nt = size(rho)

    for i=1:Nr
        rho_tmp = prob.u0
        rho[i, 1] = sum(prob.u0)
        for j=1:Nt-1
            args = (E[i, j], )

            rho_tmp = prob.step(rho_tmp, dt, args)
            rho[i, j+1] = sum(rho_tmp)

            kdrho_tmp = kdrho_term.calculate(rho_tmp, args)
            kdrho[i, j] = sum(Ks .* kdrho_tmp)   # i,j and not i,j+1 since E[i,j] -> Iarg
        end
        kdrho[i, end] = 0.
    end
    return nothing
end


function solve!(rho::CuArrays.CuArray{T},
                kdrho::CuArrays.CuArray{T},
                E::CuArrays.CuArray{Complex{T}},
                p::Tuple) where T<:AbstractFloat
    Nr, Nt = size(rho)

    # nth = Nr
    nth = 256
    nbl = Int(ceil(Nr / nth))
    @CUDAnative.cuda blocks=nbl threads=nth solve_kernel(rho, kdrho, E, p)
    return nothing
end


function solve_kernel(rho, kdrho, E, p)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x

    prob, dt, kdrho_term, Ks = p

    Nr, Nt = size(rho)

    for i=id:stride:Nr
        rho_tmp = prob.u0
        rho[i, 1] = sum(prob.u0)
        for j=1:Nt-1
            args = (E[i, j], )

            rho_tmp = prob.step(rho_tmp, dt, args)
            rho[i, j + 1] = sum(rho_tmp)

            kdrho_tmp = kdrho_term.calculate(rho_tmp, args)
            kdrho[i, j] = sum(Ks .* kdrho_tmp)   # i,j and not i,j+1 since E[i,j] -> Iarg
        end
        kdrho[i, end] = 0.
    end
    return nothing
end


end
