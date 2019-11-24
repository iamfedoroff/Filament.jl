prefix = "results/"   # the path and prefix for the output files

# Initial conditions -----------------------------------------------------------
file_initial_condition = "gauss.jl"   # file with initial conditions

# Medium -----------------------------------------------------------------------
file_medium = "air.jl"   # file with medium parameters

# Units ------------------------------------------------------------------------
zu = 1.   # [m] unit of space in z direction
tu = 1e-15   # [s] unit of time
Iu = 1e12 * 1e4   # [W/m**2] unit of intensity
rhou = 2.5e25   # [1/m**3] unit of plasma density

# Grid -------------------------------------------------------------------------
geometry = "T"   # grid geometry (R, T, RT, XY, XYT)

tmin, tmax = -100., 100.   # [tu] area in time domain
Nt = 2048   # number of points in the time domain

# Model ------------------------------------------------------------------------
zmax = 20.   # [zu] propagation distance
dz_initial = zmax / 100.   # initial z step

NONLINEARITY = false   # presence of nonlinear terms
PLASMA = false   # solve plasma equation

tguard = 10.   # [tu] the width of the lossy slab at the end of t grid
wguard = 1e16   # [1/s] the cut-off angular frequency

dzphimax = pi / 100.   # maximum nonlinear phase for adaptive z step
Istop = 1e3   # [Iu] maixmum intensity in the problem (stop if exceeded)

ALG = "RK3"   # Solver algorithm for nonlinear part ("RK2", "RK3", "RK4", "Tsit5", or "ATsit5")

# Plots ------------------------------------------------------------------------
dz_plothdf = zmax   # [zu] z step for writing the field into the HDF5 file
