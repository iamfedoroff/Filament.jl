prefix = "results/"   # the path and prefix for the output files

# Initial conditions -----------------------------------------------------------
file_initial_condition = "gauss.jl"   # file with initial conditions

# Medium -----------------------------------------------------------------------
file_medium = "air.jl"   # file with medium parameters

# Units ------------------------------------------------------------------------
ru = 1e-3   # [m] unit of space in r direction
zu = 1.   # [m] unit of space in z direction
Iu = 1e12 * 1e4   # [W/m**2] unit of intensity

# Grid -------------------------------------------------------------------------
geometry = "R"   # grid geometry (R, T, RT, XY, XYT)

rmax = 10.   # [ru] area in spatial domain
Nr = 2000  # number of points in spatial domain

# Model ------------------------------------------------------------------------
zmax = 1.   # [zu] propagation distance
dz_initial = zmax / 200.   # initial z step

KPARAXIAL = 0   # switch for the paraxial approximation of the linear term
QPARAXIAL = 1   # switch for the paraxial approximation of the nonlinear term

rguard = 0.   # [ru] the width of the lossy slab at the end of r grid
kguard = 90.   # [degrees] the cut-off angle for wave vectors

dzAdaptLevel = pi / 100.   # phase criterium for the adaptive z step
Istop = 1e3   # [Iu] maixmum intensity in the problem (stop if exceeded)

RKORDER = 4   # Order of the Runge-Kutta method (2, 3 or 4)

# Plots ------------------------------------------------------------------------
dz_plothdf = zmax / 200.   # [zu] z step for writing the field into the HDF5 file
