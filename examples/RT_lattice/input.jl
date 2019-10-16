prefix = "results/"   # the path and prefix for the output files

# Initial conditions -----------------------------------------------------------
file_initial_condition = "gauss.jl"   # file with initial conditions

# Medium -----------------------------------------------------------------------
file_medium = "air.jl"   # file with medium parameters

# Units ------------------------------------------------------------------------
ru = 1e-3   # [m] unit of space in r direction
zu = 1.   # [m] unit of space in z direction
tu = 1e-15   # [s] unit of time
Iu = 1e12 * 1e4   # [W/m**2] unit of intensity
rhou = 2.5e25   # [1/m**3] unit of plasma density

# Grid -------------------------------------------------------------------------
geometry = "RT"   # grid geometry (R, T, RT, XY, XYT)

rmax = 10.   # [ru] area in spatial domain
Nr = 500  # number of points in spatial domain

tmin, tmax = -200., 200.   # [tu] area in time domain
Nt = 2048   # number of points in the time domain

# Model ------------------------------------------------------------------------
zmax = 4.   # [zu] propagation distance
dz_initial = 0.01   # initial z step

NONLINEARITY = true   # presence of nonlinear terms
PLASMA = true   # solve plasma equation

KPARAXIAL = false   # paraxial approximation for the linear term
QPARAXIAL = true   # paraxial approximation for the nonlinear term

rguard = 1.   # [ru] the width of the lossy slab at the end of r grid
tguard = 20.   # [tu] the width of the lossy slab at the end of t grid
kguard = 45.   # [degrees] the cut-off angle for wave vectors
wguard = 1e16   # [1/s] the cut-off angular frequency

dzphimax = pi / 100.   # maximum nonlinear phase for adaptive z step
Istop = 1e3   # [Iu] maixmum intensity in the problem (stop if exceeded)

ALG = "RK3"   # Solver algorithm for nonlinearity part ("RK2", "RK3", "RK4", "Tsit5", or "ATsit5")

# Plots ------------------------------------------------------------------------
dz_plothdf = 0.1   # [zu] z step for writing the field into the HDF5 file
