module AdaptiveSteps

import ..Constants: EPS0, MU0, QE, ME
import ..FieldAnalyzers
import ..Media
import ..Units


abstract type AStep end


struct AStepKerr{T<:AbstractFloat, B<:Bool} <: AStep
    dz0 :: T
    phimax :: T
    phik :: T
    NONLINEARITY :: B
end


struct AStepKerrPlasma{T<:AbstractFloat, B<:Bool} <: AStep
    dz0 :: T
    phimax :: T
    phik :: T
    phip :: T
    NONLINEARITY :: B
end


function AStep(
    unit, medium, field, dz0::T, phimax::T, NONLINEARITY::Bool,
) where T<:AbstractFloat
    phik = phi_kerr(unit, medium, field)
    return AStepKerr(dz0, phimax, phik, NONLINEARITY)
end


function AStep(
    unit, medium, field, dz0::T, phimax::T, mr::T, nuc::T, NONLINEARITY::Bool,
) where T<:AbstractFloat
    phik = phi_kerr(unit, medium, field)
    phip = phi_plasma(unit, medium, field, mr, nuc)
    return AStepKerrPlasma(dz0, phimax, phik, phip, NONLINEARITY)
end


function phi_kerr(unit, medium, field)
    w0 = field.w0
    mu = medium.permeability(w0)
    n0 = Media.refractive_index(medium, w0)
    k0 = Media.k_func(medium, w0)
    chi3 = Media.chi3_func(medium, w0)
    Eu = Units.E(unit, real(n0))

    QZ0 = MU0 * mu * w0^2 / (2 * k0) * unit.z / Eu
    Rnl0 = EPS0 * chi3 * 3 / 4 * Eu^3
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
    QZ0 = MU0 * mu * w0^2 / (2 * k0) * unit.z / Eu
    Rnl0 = 1im / w0 * QE^2 / MR / (nuc - 1im * w0) * unit.rho * Eu
    return QZ0 * abs(real(Rnl0))
end


function (dzadaptive::AStepKerr)(analyzer::FieldAnalyzers.FieldAnalyzer)
    return dzadaptive(analyzer.Imax)
end


function (dzadaptive::AStepKerr)(Imax::T) where T<:AbstractFloat
    if dzadaptive.NONLINEARITY
        dzk = dzadaptive.phimax / (dzadaptive.phik * Imax)
        dz = min(dzadaptive.dz0, dzk)
    else
        dz = dzadaptive.dz0
    end
    return convert(T, dz)
end


function (dzadaptive::AStepKerrPlasma)(analyzer::FieldAnalyzers.FieldAnalyzer)
    return dzadaptive(analyzer.Imax, analyzer.rhomax)
end


function (dzadaptive::AStepKerrPlasma)(
    Imax::T, rhomax::T,
) where T<:AbstractFloat
    if dzadaptive.NONLINEARITY
        dzk = dzadaptive.phimax / (dzadaptive.phik * Imax)
        dzp = dzadaptive.phimax / (dzadaptive.phip * rhomax)
        dz = min(dzadaptive.dz0, dzk, dzp)
    else
        dz = dzadaptive.dz0
    end
    return convert(T, dz)
end




end
