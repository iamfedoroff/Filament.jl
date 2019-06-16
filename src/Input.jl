module Input

# Modules and variables available in input files:
import CUDAnative
import CuArrays
import CUDAdrv
import FFTW

import Fourier
import Units
import Media

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

const DEFPATH = joinpath(@__DIR__, "modules", "medium_responses")

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

keys = (KPARAXIAL = KPARAXIAL, QPARAXIAL = QPARAXIAL, ALG = ALG)

if geometry == "R"
    p_unit = (ru, zu, Iu)
    p_grid = (rmax, Nr)
    p_guard = (rguard, kguard)
    p_model = (keys, responses)
elseif geometry == "T"
    println("T geometry is not implemented yet.")
    exit()
elseif geometry == "RT"
    p_unit = (ru, zu, tu, Iu, rhou)
    p_grid = (rmax, Nr, tmin, tmax, Nt)
    p_guard = (rguard, tguard, kguard, wguard)
    p_model = (keys, responses, plasma_equation)
elseif geometry == "XY"
    p_unit = (xu, yu, zu, Iu)
    p_grid = (xmin, xmax, Nx, ymin, ymax, Ny)
    p_guard = (xguard, yguard, kxguard, kyguard)
    p_model = (keys, responses)
elseif geometry == "XYT"
    println("XYT geometry is not implemented yet.")
    exit()
else
    println("ERROR: Wrong grid geometry.")
    exit()
end

end
