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
rmax = 10.   # [ru] area in spatial domain
Nr = 500  # number of points in spatial domain

tmin, tmax = -200., 200.   # [tu] area in time domain
Nt = 2048   # number of points in the time domain

# Model ------------------------------------------------------------------------
zmax = 4.   # [zu] propagation distance
dz_initial = zmax / 100.   # initial z step

KPARAXIAL = 0   # switch for the paraxial approximation of the linear term

rguard_width = 1.   # [ru] the width of the lossy slab at the end of r grid
tguard_width = 20.   # [tu] the width of the lossy slab at the end of t grid
kguard = 45.   # [degrees] the cut-off angle for wave vectors
wguard = 1e16   # [1/s] the cut-off angular frequency

Istop = 1e4   # [Iu] maixmum intensity in the problem (stop if exceeded)

# Plots ------------------------------------------------------------------------
dz_plothdf = zmax / 100.   # [zu] z step for writing the field into the HDF5 file
