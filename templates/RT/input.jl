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
dz_initial = 0.01   # initial z step

KPARAXIAL = 0   # switch for the paraxial approximation of the linear term
QPARAXIAL = 1   # switch for the paraxial approximation of the nonlinear term

KERR = 1   # switch for Kerr nonlinearity
THG = 1   # switch for third harmonic generation

RAMAN = 1   # switch for stimulated Raman nonlinearity
RTHG = 1   # switch for third harmonic generation by stimulated Raman effect

PLASMA = 1   # switch for plasma nonlinearity
ILOSSES = 1   # switch for losses due to multiphoton ionization
IONARG = 1   # switch for the ionization rate argument: 1 - abs(E), 0 - real(E)
AVALANCHE = 1  # switch for avalanche ionization

rguard_width = 1.   # [ru] the width of the lossy slab at the end of r grid
tguard_width = 20.   # [tu] the width of the lossy slab at the end of t grid
kguard = 45.   # [degrees] the cut-off angle for wave vectors
wguard = 1e16   # [1/s] the cut-off angular frequency

dzAdaptLevel = pi / 100.   # phase criterium for the adaptive z step
Istop = 1e2   # [Iu] maixmum intensity in the problem (stop if exceeded)

# Plots ------------------------------------------------------------------------
dz_plothdf = 0.1   # [zu] z step for writing the field into the HDF5 file
