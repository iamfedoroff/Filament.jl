module NonlinearResponses

    import CUDAnative
    import CuArrays
    import CUDAdrv
    import FFTW

    import PyCall

    import Units
    import Fields
    import Media
    import Fourier

    scipy_constants = PyCall.pyimport("scipy.constants")
    const C0 = scipy_constants.c   # speed of light in vacuum
    const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
    const MU0 = scipy_constants.mu_0   # the magnetic constant [N/A^2]
    const QE = scipy_constants.e   # elementary charge [C]
    const ME = scipy_constants.m_e   # electron mass [kg]
    const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]

    const FloatGPU = Float32
    const ComplexGPU = ComplexF32


    struct NonlinearResponse{T}
        Rnl :: T
        func :: Function
        p :: Tuple
    end


    function calculate!(nresp::NonlinearResponse, F, E)
        nresp.func(F, E, nresp.p)
        return nothing
    end

end
