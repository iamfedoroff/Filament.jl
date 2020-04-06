module Input

# Modules and variables available in input files:
import CUDAnative
import CuArrays
import FFTW
import StaticArrays

import Constants: FloatGPU, MAX_THREADS_PER_BLOCK, C0, EPS0, MU0, QE, ME, HBAR
import Fourier
import Grids
import Units
import Media
import Equations
import TabulatedFunctions

const DEFPATHNR = joinpath(@__DIR__, "modules", "medium_responses")
const DEFPATHPE = joinpath(@__DIR__, "modules", "plasma_equations")

# Read input file and change current working directory:
file_input = abspath(ARGS[1])
include(file_input)
cd(dirname(file_input))

# Read initial condition file:
file_initial_condition = abspath(file_initial_condition)
include(file_initial_condition)

# Read medium file:
file_medium = abspath(file_medium)
include(file_medium)

z = z / zu   # convert initial z to dimensionless units

if ! NONLINEARITY
    responses = []
end

if geometry == "R"
    rmax = convert(FloatGPU, rmax)
    rguard = convert(FloatGPU, rguard)
    kguard = convert(FloatGPU, kguard)
    z = convert(FloatGPU, z)
    lam0 = convert(FloatGPU, lam0)
    keys = (
        NONLINEARITY=NONLINEARITY,
        KPARAXIAL=KPARAXIAL,
        QPARAXIAL=QPARAXIAL,
        ALG=ALG,
    )
    p_unit = (ru, zu, Iu)
    p_grid = (rmax, Nr)
    p_guard = (rguard, kguard)
    p_model = (responses, keys)
    p_dzadaptive = (dzphimax, )
elseif geometry == "T"
    if ! PLASMA
        plasma_equation = Dict()
    end
    keys = (
        NONLINEARITY=NONLINEARITY,
        PLASMA=PLASMA,
        ALG=ALG,
    )
    p_unit = (zu, tu, Iu, rhou)
    p_grid = (tmin, tmax, Nt)
    p_guard = (tguard, wguard)
    p_model = (responses, plasma_equation, keys)
    p_dzadaptive = (dzphimax, mr, nuc)
elseif geometry == "RT"
    rmax = convert(FloatGPU, rmax)
    tmin = convert(FloatGPU, tmin)
    tmax = convert(FloatGPU, tmax)
    rguard = convert(FloatGPU, rguard)
    tguard = convert(FloatGPU, tguard)
    kguard = convert(FloatGPU, kguard)
    wguard = convert(FloatGPU, wguard)
    z = convert(FloatGPU, z)
    lam0 = convert(FloatGPU, lam0)
    if ! PLASMA
        plasma_equation = Dict()
    end
    keys = (
        NONLINEARITY=NONLINEARITY,
        PLASMA=PLASMA,
        KPARAXIAL=KPARAXIAL,
        QPARAXIAL=QPARAXIAL,
        ALG=ALG,
    )
    p_unit = (ru, zu, tu, Iu, rhou)
    p_grid = (rmax, Nr, tmin, tmax, Nt)
    p_guard = (rguard, tguard, kguard, wguard)
    p_model = (responses, plasma_equation, keys)
    p_dzadaptive = (dzphimax, mr, nuc)
elseif geometry == "XY"
    xmin = convert(FloatGPU, xmin)
    xmax = convert(FloatGPU, xmax)
    ymin = convert(FloatGPU, ymin)
    ymax = convert(FloatGPU, ymax)
    xguard = convert(FloatGPU, xguard)
    yguard = convert(FloatGPU, yguard)
    kxguard = convert(FloatGPU, kxguard)
    kyguard = convert(FloatGPU, kyguard)
    z = convert(FloatGPU, z)
    lam0 = convert(FloatGPU, lam0)
    keys = (
        NONLINEARITY=NONLINEARITY,
        KPARAXIAL=KPARAXIAL,
        QPARAXIAL=QPARAXIAL,
        ALG=ALG,
    )
    p_unit = (xu, yu, zu, Iu)
    p_grid = (xmin, xmax, Nx, ymin, ymax, Ny)
    p_guard = (xguard, yguard, kxguard, kyguard)
    p_model = (responses, keys)
    p_dzadaptive = (dzphimax, )
elseif geometry == "XYT"
    throw(DomainError("XYT geometry is not implemented yet."))
else
    throw(DomainError("Wrong grid geometry."))
end

end
