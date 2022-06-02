# This initial condition file demonstrates how one can continue a stopped
# simulation from a specific distance
import HDF5

# HDF file with the field (calculated on the same grid):
fname = "./results/ok_plot.h5"
dset = "015"   # dataset representing the field at a specific distance

# read the field data:
pf = HDF5.h5open(fname, "r")
Edata = pf["field/" * dset]

# the distance at which the initial condition is defined:
z = HDF5.read(HDF5.attributes(Edata)["z"])
z = z * HDF5.read(pf["units/z"])   # convert to SI units

lam0 = 0.8e-6   # [m] central wavelength


function initial_condition(r, t, ru, tu, Iu)
    return HDF5.read(Edata)
end
