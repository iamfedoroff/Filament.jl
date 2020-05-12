module Constants

import CUDAdrv
import CUDAnative
using PhysicalConstants.CODATA2018

# const TFloat = Float32
const FloatGPU = Float32

# const A = convert(TFloat, sc.A)
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const MU0 = VacuumMagneticPermeability.val
const QE = ElementaryCharge.val
const ME = ElectronMass.val
const HBAR = ReducedPlanckConstant.val

const MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(
    CUDAnative.CuDevice(0), CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
)

end
