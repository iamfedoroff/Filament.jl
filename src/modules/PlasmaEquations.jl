module PlasmaEquations

import CUDAnative
import CuArrays
import CUDAdrv

import Units
import Grids
import Fields
import Media
import RungeKuttas
import TabularFunctions

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const QE = scipy_constants.e   # elementary charge [C]
const ME = scipy_constants.m_e   # electron mass [kg]

const FloatGPU = Float32


struct Component{T<:AbstractFloat}
    name :: String
    frho0 :: T
    K :: T
    Rava :: T
    Rgamma :: T
    tabfunc :: TabularFunctions.TabularFunction
end


function Component(unit::Units.Unit, field::Fields.Field, medium::Media.Medium,
                   rho0::T, nuc::T, mr::T, name::String, frac::T, Ui::T,
                   fname_tabfunc::String) where T<:AbstractFloat
    frho0 = frac * rho0 / unit.rho
    frho0 = FloatGPU(frho0)

    Ui = Ui * QE   # eV -> J

    Wph = Fields.energy_photon(field)
    K = ceil(Ui / Wph)   # order of multiphoton ionization
    K = FloatGPU(K)

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    MR = mr * ME   # reduced mass of electron and hole (effective mass)
    sigma = QE^2 / MR * nuc / (nuc^2 + field.w0^2)
    Rava = sigma / Ui * Eu^2 * unit.t
    Rava = FloatGPU(Rava)

    Rgamma = field.w0 / QE * sqrt(MR * n0 * EPS0 * C0 * Ui / unit.I)
    Rgamma = FloatGPU(Rgamma)

    tabfunc = TabularFunctions.TabularFunction(FloatGPU, unit, fname_tabfunc)

    return Component(name, frho0, K, Rava, Rgamma, tabfunc)
end


struct PlasmaEquation
    probs :: Array{NamedTuple}
    dt :: AbstractFloat
    fearg :: Function
    kdrho_calc :: Function
    kdrho_params :: Array{Tuple}
    nthreads :: Int
    nblocks :: Int
end


function PlasmaEquation(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                        medium::Media.Medium, args::Dict)
    dt = FloatGPU(grid.dt)

    alg = args["ALG"]
    @assert alg in ("RK2", "RK3", "RK4")

    AVALANCHE = args["AVALANCHE"]
    if AVALANCHE
        stepfunc = stepfunc_field_avalanche
    else
        stepfunc = stepfunc_field
    end

    KGAMMA = args["KGAMMA"]
    if KGAMMA
        kdrho_calc = kdrho_func_Kgamma
    else
        kdrho_calc = kdrho_func
    end

    EREAL = args["EREAL"]
    fearg_real(x::Complex{FloatGPU}) = real(x)^2
    fearg_abs2(x::Complex{FloatGPU}) = abs2(x)
    if EREAL
        fearg = fearg_real
    else
        fearg = fearg_abs2
    end

    rho0 = args["rho0"]
    nuc = args["nuc"]
    mr = args["mr"]
    components_dict = args["components"]
    Ncomp = length(components_dict)
    probs = Array{NamedTuple}(undef, Ncomp)
    kdrho_params = Array{Tuple}(undef, Ncomp)
    for i=1:Ncomp
        comp_dict = components_dict[i]
        name = comp_dict["name"]
        frac = comp_dict["fraction"]
        Ui = comp_dict["ionization_energy"]
        fname_tabfunc = comp_dict["tabular_function"]
        comp = Component(unit, field, medium, rho0, nuc, mr, name, frac, Ui, fname_tabfunc)

        p = (comp.tabfunc, comp.frho0, comp.Rava)
        probs[i] = RungeKuttas.Problem(alg, FloatGPU(0), stepfunc, p)

        kdrho_params[i] = (comp.tabfunc, comp.frho0, comp.K, comp.Rgamma)
    end

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreads = min(grid.Nr, MAX_THREADS_PER_BLOCK)
    nblocks = Int(ceil(grid.Nr / nthreads))

    PlasmaEquation(probs, dt, fearg, kdrho_calc, kdrho_params, nthreads, nblocks)
end


function solve!(PE::PlasmaEquation,
                rho::AbstractArray{T, 2},
                kdrho::AbstractArray{T, 2},
                E::AbstractArray{Complex{T}, 2}) where T<:AbstractFloat
    Nr, Nt = size(rho)
    nth = PE.nthreads
    nbl = PE.nblocks

    fill!(rho, convert(T, 0))
    fill!(kdrho, convert(T, 0))

    for i=1:length(PE.probs)
        prob = PE.probs[i]
        kdrho_p = PE.kdrho_params[i]
        @CUDAnative.cuda blocks=nbl threads=nth solve_kernel(rho, kdrho, PE.dt, PE.fearg, E, prob, PE.kdrho_calc, kdrho_p)
        CUDAdrv.synchronize()
    end
    return nothing
end


function solve_kernel(rho, kdrho, dt, fearg, E, prob, kdrho_calc, kdrho_p)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nx, Nt = size(rho)
    for i=id:stride:Nx
        tmp = prob.u0
        rho[i, 1] = rho[i, 1] + tmp
        for j=1:Nt-1
            Iarg = fearg(E[i, j])
            args = (Iarg, )
            kdrho[i, j] = kdrho[i, j] + kdrho_calc(tmp, kdrho_p, args)   # i,j and not i,j+1 since E[i,j] -> Iarg
            tmp = RungeKuttas.step(prob, tmp, dt, args)
            rho[i, j + 1] = rho[i, j + 1] + tmp
        end
        kdrho[i, end] = FloatGPU(0)
    end
    return nothing
end


function stepfunc_field(rho::AbstractFloat, p::Tuple, args::Tuple)
    tabfunc, frho0 = p
    I, = args
    R1 = tabfunc(I)
    return R1 * (frho0 - rho)
end


function stepfunc_field_avalanche(rho::AbstractFloat, p::Tuple, args::Tuple)
    tabfunc, frho0, Rava = p
    I, = args
    R1 = tabfunc(I)
    R2 = Rava * I
    return R1 * (frho0 - rho) + R2 * rho
end


function kdrho_func(rho::AbstractFloat, p::Tuple, args::Tuple)
    tabfunc, frho0, K = p
    I, = args
    R1 = tabfunc(I)
    drho = R1 * (frho0 - rho)
    return K * drho
end


function kdrho_func_Kgamma(rho::AbstractFloat, p::Tuple, args::Tuple)
    tabfunc, frho0, K, Rgamma = p
    I, = args
    # drho:
    R1 = tabfunc(I)
    drho = R1 * (frho0 - rho)
    # Kgamma:
    gamma = Rgamma / CUDAnative.sqrt(I)
    Kgamma = Kgamma_func(gamma, K)
    return Kgamma * drho
end


"""
Calculate the dependence of number of photons, K, needed to ionize one atom, on
Keldysh parameter gamma.
"""
function Kgamma_func(x::T, K::T) where T<:AbstractFloat
    # Keldysh gamma which defines the boundary between multiphoton and tunnel
    # ionization regimes:
    gamma0 = 1.5
    p = 6   # order of supergaussians used for the transition function
    if x >= 2 * gamma0
        Kgamma = K
    else
        gauss = 1 - CUDAnative.exp(-CUDAnative.pow(x / gamma0, p)) +
                CUDAnative.exp(-CUDAnative.pow((x - 2 * gamma0) / gamma0, p))
        Kgamma = 1 + (K - 1) * 0.5 * gauss
    end
    return Kgamma
end


end
