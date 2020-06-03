module Constants

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

end
