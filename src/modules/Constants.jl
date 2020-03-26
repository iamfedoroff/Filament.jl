module Constants

import CUDAdrv
import CUDAnative

import PyCall
sc = PyCall.pyimport("scipy.constants")

# const TFloat = Float32
const FloatGPU = Float32

# const A = convert(TFloat, sc.A)
const C0 = sc.c   # speed of light in vacuum
const EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = sc.mu_0   # the magnetic constant [N/A^2]
const QE = sc.e   # elementary charge [C]
const ME = sc.m_e   # electron mass [kg]
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]

const MAX_THREADS_PER_BLOCK =
        CUDAdrv.attribute(
            CUDAnative.CuDevice(0),
            CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )

end
