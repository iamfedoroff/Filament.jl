module Plasmas

import CUDAnative
import CuArrays
import CUDAdrv

import Units
import Grids
import Fields
import Media
import PlasmaComponents

const FloatGPU = Float32
const ComplexGPU = ComplexF32


struct Plasma
    nuc :: Float64
    mr :: Float64
    components :: Array{PlasmaComponents.Component, 1}
    Ncomp :: Int64

    rho_end :: Array{Float64, 1}

    rho :: CuArrays.CuArray{FloatGPU, 2}
    Kdrho :: CuArrays.CuArray{FloatGPU, 2}
    RI :: CuArrays.CuArray{FloatGPU, 2}

    rho_comp :: CuArrays.CuArray{FloatGPU, 2}
    Kdrho_comp :: CuArrays.CuArray{FloatGPU, 2}
    RI_comp :: CuArrays.CuArray{FloatGPU, 2}

    frho0s :: CuArrays.CuArray{FloatGPU, 1}
    Ks :: CuArrays.CuArray{FloatGPU, 1}
    Wavas :: CuArrays.CuArray{FloatGPU, 1}
    tfxs :: CuArrays.CuArray{FloatGPU, 2}
    tfys :: CuArrays.CuArray{FloatGPU, 2}

    IONARG :: Int64
    AVALANCHE :: Int64

    nblocks :: Int64
    nthreads :: Int64
end


function Plasma(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                medium::Media.Medium, rho0::Float64, nuc::Float64, mr::Float64,
                components_dict::Array{Dict{String, Any}, 1},
                keys)

    Ncomp = length(components_dict)
    components = Array{PlasmaComponents.Component}(undef, Ncomp)
    for i=1:Ncomp
        comp_dict = components_dict[i]
        name = comp_dict["name"]
        frac = comp_dict["fraction"]
        Ui = comp_dict["ionization_energy"]
        fname_tabfunc = comp_dict["tabular_function"]
        components[i] = PlasmaComponents.Component(unit, grid, field, medium,
                                                   rho0, nuc, mr, name, frac,
                                                   Ui, fname_tabfunc, keys)
    end

    rho_end = zeros(grid.Nr)

    rho = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    Kdrho = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    RI = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))

    rho_comp = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    Kdrho_comp = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))
    RI_comp = CuArrays.cuzeros(FloatGPU, (grid.Nr, grid.Nt))

    frho0s = zeros(Ncomp)
    Ks = zeros(Ncomp)
    Wavas = zeros(Ncomp)
    tfxs = zeros((Ncomp, length(components[1].tf.x)))
    tfys = zeros((Ncomp, length(components[1].tf.y)))
    for i=1:Ncomp
        frho0s[i] = components[i].frho0
        Ks[i] = components[i].K
        Wavas[i] = components[i].Wava
        @. tfxs[i, :] = components[i].tf.x
        @. tfys[i, :] = components[i].tf.y
    end
    frho0s = CuArrays.CuArray(convert(Array{FloatGPU, 1}, frho0s))
    Ks = CuArrays.CuArray(convert(Array{FloatGPU, 1}, Ks))
    Wavas = CuArrays.CuArray(convert(Array{FloatGPU, 1}, Wavas))
    tfxs = CuArrays.CuArray(convert(Array{FloatGPU, 2}, tfxs))
    tfys = CuArrays.CuArray(convert(Array{FloatGPU, 2}, tfys))

    IONARG = keys["IONARG"]
    AVALANCHE = keys["AVALANCHE"]

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    # nthreads = min(grid.Nr, MAX_THREADS_PER_BLOCK)   # CUDA error: too many resources requested for launch
    nthreads = 256
    nblocks = Int(ceil(grid.Nr / nthreads))

    return Plasma(nuc, mr, components, Ncomp,
                  rho_end, rho, Kdrho, RI, rho_comp, Kdrho_comp, RI_comp,
                  frho0s, Ks, Wavas, tfxs, tfys, IONARG, AVALANCHE,
                  nblocks, nthreads)
end


function peak_plasma_density(plasma::Plasma)
    return maximum(plasma.rho_end)
end


function plasma_radius(grid::Grids.Grid, plasma::Plasma)
    # Factor 2. because rho(r) is only half of full distribution rho(x)
    return 2. * Fields.radius(grid.r, plasma.rho_end)
end


"""
Linear plasma density:
    lrho = Int[rho * 2*pi*r*dr],   [lrho] = 1/m
"""
function linear_plasma_density(grid::Grids.Grid, plasma::Plasma)
    lrho = 0.
    for i=1:grid.Nr
        lrho = lrho + plasma.rho_end[i] * grid.r[i] * grid.dr[i]
    end
    return lrho * 2. * pi
end


function free_charge(plasma::Plasma, grid::Grids.Grid, field::Fields.Field)
    nbl = plasma.nblocks
    nth = plasma.nthreads
    @CUDAnative.cuda blocks=nbl threads=nth kernel(plasma.rho, plasma.Kdrho,
                                                   plasma.RI, plasma.rho_comp,
                                                   plasma.Kdrho_comp,
                                                   plasma.RI_comp, field.E_gpu,
                                                   grid.dt, plasma.frho0s,
                                                   plasma.Ks, plasma.Wavas,
                                                   plasma.tfxs, plasma.tfys,
                                                   plasma.IONARG,
                                                   plasma.AVALANCHE)
    plasma.rho_end[:] = CuArrays.collect(plasma.rho[:, end])
    return nothing
end


function kernel(rho, Kdrho, RI, rho_comp, Kdrho_comp, RI_comp, E, dt, frho0s,
                Ks, Wavas, tfxs, tfys, IONARG, AVALANCHE)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(rho)
    Ncomp, Ntf = size(tfxs)

    for i=id:stride:Nr
        for j=1:Nt
            rho[i, j] = FloatGPU(0)
            Kdrho[i, j] = FloatGPU(0)
            RI[i, j] = FloatGPU(0)
        end
    end

    for n=1:Ncomp
        frho0 = frho0s[n]
        K = Ks[n]
        Wava = Wavas[n]

        for i=id:stride:Nr
            rho_comp[i, 1] = FloatGPU(0)
            Kdrho_comp[i, 1] = FloatGPU(0)
            RI_comp[i, 1] = FloatGPU(0)

            for j=2:Nt
                if IONARG != 0
                    Ival = FloatGPU(0.5) * (abs2(E[i, j]) + abs2(E[i, j-1]))
                else
                    Ival = FloatGPU(0.5) * (real(E[i, j])^2 + real(E[i, j-1])^2)
                end

                xc = CUDAnative.log10(Ival)
                if xc < tfxs[n, 1]
                    W1 = FloatGPU(0)
                elseif xc >= tfxs[n, end]
                    W1 = CUDAnative.pow(FloatGPU(10.), tfys[n, end])
                else
                    xcnorm = (xc - tfxs[n, 1]) / (tfxs[n, end] - tfxs[n, 1])
                    iloc = Int32(CUDAnative.floor(xcnorm * Ntf + 1.))
                    W1log = tfys[n, iloc] + (tfys[n, iloc + 1] - tfys[n, iloc]) * (xc - tfxs[n, iloc]) /
                                            (tfxs[n, iloc + 1] - tfxs[n, iloc])
                    W1 = CUDAnative.pow(FloatGPU(10), W1log)
                end

                if AVALANCHE != 0
                    W2 = Wava * Ival
                else
                    W2 = FloatGPU(0)
                end

                if W1 == 0.
                    # if no field ionization, then calculate only the avalanche one
                    rho_comp[i, j] = rho_comp[i, j-1] * CUDAnative.exp(W2 * dt)
                    Kdrho_comp[i, j] = 0.
                else
                    # without this "if" statement 1/W12 will cause NaN values in rho
                    W12 = W1 - W2
                    rho_comp[i, j] = W1 / W12 * frho0 -
                                     (W1 / W12 * frho0 - rho_comp[i, j-1]) *
                                     CUDAnative.exp(-W12 * dt)
                    Kdrho_comp[i, j] = K * W1 * (frho0 - rho_comp[i, j])
                end

                RI_comp[i, j] = W1
            end

            for j=1:Nt
                rho[i, j] = rho[i, j] + rho_comp[i, j]
                Kdrho[i, j] = Kdrho[i, j] + Kdrho_comp[i, j]
                RI[i, j] = RI[i, j] + RI_comp[i, j]
            end
        end
    end

    return nothing
end


end
