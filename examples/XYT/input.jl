prefix = "results/"   # the path and prefix for the output files

# Initial conditions -----------------------------------------------------------
file_initial_condition = "gauss.jl"   # file with initial conditions

# Medium -----------------------------------------------------------------------
file_medium = "air.jl"   # file with medium parameters

# Units ------------------------------------------------------------------------
xu = 1e-3   # [m] unit of space in x direction
yu = 1e-3   # [m] unit of space in y direction
zu = 1.0   # [m] unit of space in z direction
tu = 1e-15   # [s] unit of time
Iu = 1e12 * 1e4   # [W/m**2] unit of intensity
rhou = 2.5e25   # [1/m**3] unit of plasma density

# Grid -------------------------------------------------------------------------
geometry = "XYT"   # grid geometry (R, T, RT, XY, XYT)

xmin, xmax = -10.0, 10.0   # [xu] area in x spatial domain
Nx = 500  # number of points in x spatial domain

ymin, ymax = -10.0, 10.0   # [yu] area in y spatial domain
Ny = 500  # number of points in y spatial domain

tmin, tmax = -200.0, 200.0   # [tu] area in time domain
Nt = 1024   # number of points in the time domain

# Model ------------------------------------------------------------------------
zmax = 4.0   # [zu] propagation distance
dz_initial = 0.02   # initial z step

NONLINEARITY = false   # presence of nonlinear terms
PLASMA = false   # solve plasma equation

KPARAXIAL = false   # paraxial approximation for the linear term
QPARAXIAL = true   # paraxial approximation for the nonlinear term

xguard = 1.0   # [xu] the width of the lossy slab at the end of x grid
yguard = 1.0   # [xu] the width of the lossy slab at the end of y grid
tguard = 20.0   # [tu] the width of the lossy slab at the end of t grid
kxguard = 45.0   # [degrees] the cut-off angle for wave vectors in x grid
kyguard = 45.0   # [degrees] the cut-off angle for wave vectors in y grid
wguard = 1e16   # [1/s] the cut-off angular frequency

dzphimax = pi / 100.0   # maximum nonlinear phase for adaptive z step
Istop = 1e3   # [Iu] maixmum intensity in the problem (stop if exceeded)

ALG = "RK3"   # Solver algorithm for nonlinear part ("RK2", "RK3", "RK4", "Tsit5", or "ATsit5")

# Plots ------------------------------------------------------------------------
dz_plothdf = 1.0   # [zu] z step for writing the field into the HDF5 file
