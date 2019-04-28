prefix = "results/"   # the path and prefix for the output files

# Initial conditions -----------------------------------------------------------
file_initial_condition = "gauss.jl"   # file with initial conditions

# Medium -----------------------------------------------------------------------
file_medium = "air.jl"   # file with medium parameters

# Units ------------------------------------------------------------------------
xu = 1e-3   # [m] unit of space in x direction
yu = 1e-3   # [m] unit of space in y direction
zu = 1.   # [m] unit of space in z direction
Iu = 1e12 * 1e4   # [W/m**2] unit of intensity

# Grid -------------------------------------------------------------------------
geometry = "XY"   # grid geometry (R, T, RT, XY, XYT)

xmin, xmax = -10., 10.   # [xu] area in x spatial domain
Nx = 2048  # number of points in x spatial domain

ymin, ymax = -10., 10.   # [yu] area in y spatial domain
Ny = 2048  # number of points in y spatial domain

# Model ------------------------------------------------------------------------
zmax = 1.   # [zu] propagation distance
dz_initial = zmax / 200.   # initial z step

KPARAXIAL = 0   # switch for the paraxial approximation of the linear term
QPARAXIAL = 1   # switch for the paraxial approximation of the nonlinear term

xguard = 0.   # [xu] the width of the lossy slab at the end of x grid
yguard = 0.   # [xu] the width of the lossy slab at the end of y grid
kxguard = 90.   # [degrees] the cut-off angle for wave vectors in x grid
kyguard = 90.   # [degrees] the cut-off angle for wave vectors in y grid

dzAdaptLevel = pi / 100.   # phase criterium for the adaptive z step
Istop = 1e3   # [Iu] maixmum intensity in the problem (stop if exceeded)

RKORDER = 4   # Order of the Runge-Kutta method (2, 3 or 4)

# Plots ------------------------------------------------------------------------
dz_plothdf = zmax / 200.   # [zu] z step for writing the field into the HDF5 file
