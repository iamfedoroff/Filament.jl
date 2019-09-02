# This initial condition file demonstrates how one can continue a stopped
# simulation from a specific distance
import HDF5

# HDF file with the field (calculated on the same grid):
fname = "./results/plot.h5"
dset = "001"   # dataset representing the field at a specific distance

# read the field data:
pf = HDF5.h5open(fname, "r")
Edata = pf["field/" * dset]

# the distance at which the initial condition is defined:
z = HDF5.read(HDF5.a_open(Edata, "z"))
z = z * HDF5.read(pf["units/z"])   # convert to SI units

lam0 = 800e-9   # [m] central wavelength


function initial_condition(r, t, ru, tu, Iu)
    E = HDF5.read(Edata)
    return transpose(E)
end
