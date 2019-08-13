module AdaptiveSteps

import Units
import Media

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = scipy_constants.mu_0   # the magnetic constant [N/A^2]
const QE = scipy_constants.e   # elementary charge [C]
const ME = scipy_constants.m_e   # electron mass [kg]


abstract type AStep end


struct AStepKerr{T<:AbstractFloat} <: AStep
    phimax :: T
    phik :: T
end

struct AStepKerrPlasma{T<:AbstractFloat} <: AStep
    phimax :: T
    phik :: T
    phip :: T
end


function AStep(unit, medium, field, phimax::T) where T<:AbstractFloat
    phik = phi_kerr(unit, medium, field)
    return AStepKerr(phimax, phik)
end


function AStep(unit, medium, field, phimax::T,
               mr::T, nuc::T) where T<:AbstractFloat
    phik = phi_kerr(unit, medium, field)
    phip = phi_plasma(unit, medium, field, mr, nuc)
    return AStepKerrPlasma(phimax, phik, phip)
end


function phi_kerr(unit, medium, field)
    w0 = field.w0
    mu = medium.permeability(w0)
    n0 = Media.refractive_index(medium, w0)
    k0 = Media.k_func(medium, w0)
    chi3 = Media.chi3_func(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ0 = MU0 * mu * w0^2 / (2. * k0) * unit.z / Eu
    Rnl0 = EPS0 * chi3 * 3. / 4. * Eu^3
    return QZ0 * abs(Rnl0)
end


function phi_plasma(unit, medium, field, mr::T, nuc::T) where T<:AbstractFloat
    w0 = field.w0
    mu = medium.permeability(w0)
    n0 = Media.refractive_index(medium, w0)
    k0 = Media.k_func(medium, w0)
    MR = mr * ME   # reduced mass of electron and hole (effective mass)
    Eu = Units.E(unit, real(n0))

    MR = mr * ME   # reduced mass of electron and hole (effective mass)
    QZ0 = MU0 * mu * w0^2 / (2. * k0) * unit.z / Eu
    Rnl0 = 1im / w0 * QE^2 / MR / (nuc - 1im * w0) * unit.rho * Eu
    return QZ0 * abs(real(Rnl0))
end


function (dz::AStepKerr)(Imax::AbstractFloat)
    return dz.phimax / (dz.phik * Imax)
end


function (dz::AStepKerrPlasma)(Imax::T, rhomax::T) where T<:AbstractFloat
    dzk = dz.phimax / (dz.phik * Imax)
    dzp = dz.phimax / (dz.phip * rhomax)
    return min(dzk, dzp)
end


end
