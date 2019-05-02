module PlasmaEquations

import CUDAnative
import CuArrays
import CUDAdrv

import Units
import Grids
import Fields
import Media
import TabularFunctions

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const QE = scipy_constants.e   # elementary charge [C]
const ME = scipy_constants.m_e   # electron mass [kg]

const FloatGPU = Float32


struct Component{T<:AbstractFloat}
    name :: String
    frho0 :: T
    K :: T
    Rava :: T
    tf :: TabularFunctions.TabularFunction
end


function Component(unit::Units.Unit, field::Fields.Field, medium::Media.Medium,
                   rho0::T, nuc::T, mr::T, name::String, frac::T, Ui::T,
                   fname_tabfunc::String) where T<:AbstractFloat
    rho0 = rho0 / unit.rho
    frho0 = frac * rho0
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

    tf = TabularFunctions.CuTabularFunction(FloatGPU, unit, fname_tabfunc)

    return Component(name, frho0, K, Rava, tf)
end


struct PlasmaEquation{T<:AbstractFloat}
    dt :: T
    method :: Function
    calc :: Function
    fearg :: Function
    components :: Array{Component, 1}
    rho :: CuArrays.CuArray{T}
    drho :: CuArrays.CuArray{T}
    nthreads :: Int
    nblocks :: Int
end


function PlasmaEquation(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                        medium::Media.Medium, args::Dict)
    dt = FloatGPU(grid.dt)

    METHOD = args["METHOD"]
    if METHOD == "ETD"
        method = etd
    elseif METHOD == "RK2"
        method = rk2
    elseif METHOD == "RK3"
        method = rk3
    elseif METHOD == "RK4"
        method = rk4
    else
        println("ERROR: Wrong numerical method for kinetic equation.")
        exit()
    end

    AVALANCHE = args["AVALANCHE"]
    if AVALANCHE
        if METHOD == "ETD"
            calc = etd_field_avalanche
        else
            calc = rk_field_avalanche
        end
    else
        if METHOD == "ETD"
            calc = etd_field
        else
            calc = rk_field
        end
    end

    EREAL = args["EREAL"]
    fearg = if EREAL
        function(x::Complex{FloatGPU})
            real(x)^2
        end
    else
        function(x::Complex{FloatGPU})
            abs2(x)
        end
    end

    rho0 = args["rho0"]
    nuc = args["nuc"]
    mr = args["mr"]
    components_dict = args["components"]
    Ncomp = length(components_dict)
    components = Array{Component}(undef, Ncomp)
    for i=1:Ncomp
        comp_dict = components_dict[i]
        name = comp_dict["name"]
        frac = comp_dict["fraction"]
        Ui = comp_dict["ionization_energy"]
        fname_tabfunc = comp_dict["tabular_function"]
        components[i] = Component(unit, field, medium, rho0, nuc, mr, name,
                                  frac, Ui, fname_tabfunc)
    end

    rho = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    drho = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreads = min(grid.Nr, MAX_THREADS_PER_BLOCK)
    nblocks = Int(ceil(grid.Nr / nthreads))

    return PlasmaEquation(dt, method, calc, fearg, components, rho, drho,
                           nthreads, nblocks)
end


function solve!(PE::PlasmaEquation,
                rho::CuArrays.CuArray{T}, Kdrho::CuArrays.CuArray{T},
                E::CuArrays.CuArray{Complex{T}}) where T<:AbstractFloat
    nth = PE.nthreads
    nbl = PE.nblocks

    fill!(rho, convert(T, 0))
    fill!(Kdrho, convert(T, 0))

    for comp in PE.components
        p = (comp.tf.x, comp.tf.y, comp.tf.dy, comp.frho0, comp.Rava)
        @CUDAnative.cuda blocks=nbl threads=nth solve_kernel(PE.rho, PE.drho,
                                                             PE.dt, PE.method,
                                                             PE.calc, PE.fearg,
                                                             E, p)
        CUDAdrv.synchronize()
        @. rho = rho + PE.rho
        @. Kdrho = Kdrho + comp.K * PE.drho
        CUDAdrv.synchronize()
    end
end


function solve_kernel(rho::AbstractArray{T, 2}, drho::AbstractArray{T, 2},
                      dt::T, method::Function, calc::Function, fearg::Function,
                      E::AbstractArray{Complex{T}, 2},
                      p::Tuple) where T<:AbstractFloat
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(rho)
    for i=id:stride:N1
        rho[i, 1] = convert(T, 0)
        for j=1:N2-1
            Iarg = fearg(E[i, j])
            rho[i, j + 1] = method(rho[i, j], dt, calc, Iarg, p)
            drho[i, j] = drho_func(rho[i, j], Iarg, p)
        end
        drho[i, end] = convert(T, 0)
    end
    return nothing
end


function etd(u::T, h::T, func::Function, I::T, p::Tuple) where T<:AbstractFloat
    return func(u, h, I, p)
end


function etd_field(rho::T, dt::T, I::T, p::Tuple) where T<:AbstractFloat
    tfx = p[1]
    tfy = p[2]
    tfdy = p[3]
    frho0 = p[4]
    R1 = TabularFunctions.tfvalue(I, tfx, tfy, tfdy)
    return frho0 - (frho0 - rho) * CUDAnative.exp(-R1 * dt)
end


function etd_field_avalanche(rho::T, dt::T, I::T, p::Tuple) where T<:AbstractFloat
    tfx = p[1]
    tfy = p[2]
    tfdy = p[3]
    frho0 = p[4]
    Rava = p[5]
    R1 = TabularFunctions.tfvalue(I, tfx, tfy, tfdy)
    R2 = Rava * I
    if iszero(R1)
        # if no field ionization, then calculate only the avalanche one
        res = rho * CUDAnative.exp(R2 * dt)
    else
        dR = R1 - R2
        res =  R1 / dR * frho0 - (R1 / dR * frho0 - rho) * CUDAnative.exp(-dR * dt)
    end
    return res
end


function rk2(u::T, h::T, func::Function, I::T, p::Tuple) where T<:AbstractFloat
    k1 = func(u, I, p)

    tmp = u + h * convert(T, 2 / 3) * k1
    k2 = func(tmp, I, p)

    return u + h / convert(T, 4) * (k1 + convert(T, 3) * k2)
end


function rk3(u::T, h::T, func::Function, E::T, p::Tuple) where T<:AbstractFloat
    k1 = func(u, I, p)

    tmp = u + h * convert(T, 0.5) * k1
    k2 = func(tmp, I, p)

    tmp = u + h * (-convert(T, 1) * k1 + convert(T, 2) * k2)
    k3 = func(tmp, I, p)

    return u + h / convert(T, 6) * (k1 + convert(T, 4) * k2 + k3)
end


function rk4(u::T, h::T, func::Function, I::T, p::Tuple) where T<:AbstractFloat
    k1 = func(u, I, p)

    tmp = u + h * convert(T, 0.5) * k1
    k2 = func(tmp, I, p)

    tmp = u + h * convert(T, 0.5) * k2
    k3 = func(tmp, I, p)

    tmp = u + h * k3
    k4 = func(tmp, I, p)
    return u + h / convert(T, 6) * (k1 + convert(T, 2) * k2 +
                                         convert(T, 2) * k3 + k4)
end


function rk_field(rho::T, I::T, p::Tuple) where T<:AbstractFloat
    tfx = p[1]
    tfy = p[2]
    tfdy = p[3]
    frho0 = p[4]
    R1 = TabularFunctions.tfvalue(I, tfx, tfy, tfdy)
    return R1 * (frho0 - rho)
end


function rk_field_avalanche(rho::T, I::T, p::Tuple) where T<:AbstractFloat
    tfx = p[1]
    tfy = p[2]
    tfdy = p[3]
    frho0 = p[4]
    Rava = p[5]
    R1 = TabularFunctions.tfvalue(I, tfx, tfy, tfdy)
    R2 = Rava * I
    return R1 * (frho0 - rho) + R2 * rho
end


function drho_func(rho::T, I::T, p::Tuple) where T<:AbstractFloat
    tfx = p[1]
    tfy = p[2]
    tfdy = p[3]
    frho0 = p[4]
    R1 = TabularFunctions.tfvalue(I, tfx, tfy, tfdy)
    return R1 * (frho0 - rho)
end


end
