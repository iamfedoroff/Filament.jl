module Input

# Modules and variables available in input files:
import CUDAnative
import CuArrays
import CUDAdrv
import FFTW
import StaticArrays

import Fourier
import Units
import Media
import Equations
import TabulatedFunctions

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = scipy_constants.mu_0   # the magnetic constant [N/A^2]
const QE = scipy_constants.e   # elementary charge [C]
const ME = scipy_constants.m_e   # electron mass [kg]
const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32
const ComplexGPU = ComplexF32

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

if geometry == "T"
    keys = (NONLINEARITY=NONLINEARITY, ALG=ALG)
else
    keys = (NONLINEARITY=NONLINEARITY, KPARAXIAL=KPARAXIAL, QPARAXIAL=QPARAXIAL,
            ALG=ALG)
end

p_field = (lam0, initial_condition)

if geometry == "R"
    p_unit = (ru, zu, Iu)
    p_grid = (rmax, Nr)
    p_guard = (rguard, kguard)
    p_model = (keys, responses)
    p_dzadaptive = (dzphimax, )
elseif geometry == "T"
    p_unit = (zu, tu, Iu, rhou)
    p_grid = (tmin, tmax, Nt)
    p_guard = (tguard, wguard)
    p_model = (keys, responses, plasma_equation)
    p_dzadaptive = (dzphimax, mr, nuc)
elseif geometry == "RT"
    p_unit = (ru, zu, tu, Iu, rhou)
    p_grid = (rmax, Nr, tmin, tmax, Nt)
    p_guard = (rguard, tguard, kguard, wguard)
    p_model = (keys, responses, plasma_equation)
    p_dzadaptive = (dzphimax, mr, nuc)
elseif geometry == "XY"
    p_unit = (xu, yu, zu, Iu)
    p_grid = (xmin, xmax, Nx, ymin, ymax, Ny)
    p_guard = (xguard, yguard, kxguard, kyguard)
    p_model = (keys, responses)
    p_dzadaptive = (dzphimax, )
elseif geometry == "XYT"
    throw(DomainError("XYT geometry is not implemented yet."))
else
    throw(DomainError("Wrong grid geometry."))
end

end
