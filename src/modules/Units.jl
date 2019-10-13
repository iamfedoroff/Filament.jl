module Units

import PyCall

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]


abstract type Unit{T<:AbstractFloat} end


struct UnitR{T} <: Unit{T}
    r :: T
    k :: T
    z :: T
    I :: T
end


struct UnitT{T} <: Unit{T}
    z :: T
    t :: T
    w :: T
    lam :: T
    I :: T
    rho :: T
end


struct UnitRT{T} <: Unit{T}
    r :: T
    k :: T
    z :: T
    t :: T
    w :: T
    lam :: T
    I :: T
    rho :: T
end


struct UnitXY{T} <: Unit{T}
    x :: T
    y :: T
    kx :: T
    ky :: T
    z :: T
    I :: T
end


function Unit(geometry::String, p::Tuple)
    if geometry == "R"
        unit = UnitR(p...)
    elseif geometry == "T"
        unit = UnitT(p...)
    elseif geometry == "RT"
        unit = UnitRT(p...)
    elseif geometry == "XY"
        unit = UnitXY(p...)
    elseif geometry == "XYT"
        throw(DomainError("XYT geometry is not implemented yet."))
    else
        throw(DomainError("Wrong grid geometry."))
    end
    return unit
end


function UnitR(ru::T, zu::T, Iu::T) where T<:AbstractFloat
    ku = 1 / ru
    return UnitR(ru, ku, zu, Iu)
end


function UnitT(zu::T, tu::T, Iu::T, rhou::T) where T<:AbstractFloat
    wu = 1 / tu
    lamu = tu
    return UnitT(zu, tu, wu, lamu, Iu, rhou)
end


function UnitRT(ru::T, zu::T, tu::T, Iu::T, rhou::T) where T<:AbstractFloat
    ku = 1 / ru
    wu = 1 / tu
    lamu = tu
    return UnitRT(ru, ku, zu, tu, wu, lamu, Iu, rhou)
end


function UnitXY(xu::T, yu::T, zu::T, Iu::T) where T<:AbstractFloat
    kxu = 1 / xu
    kyu = 1 / yu
    return UnitXY(xu, yu, kxu, kyu, zu, Iu)
end


"""
Units of electric field (depends on refractive index n)
"""
function E(unit::Unit, n::Union{AbstractFloat, Complex})
    Eu = sqrt(unit.I / (0.5 * real(n) * EPS0 * C0))
    return convert(typeof(unit.I), Eu)
end


end
