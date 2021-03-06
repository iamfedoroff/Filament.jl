module Units

import ..Constants: C0, EPS0


abstract type Unit{T} end


function Unit(geometry::String, p::Tuple)
    if (geometry == "R") | (geometry == "Rn")
        unit = UnitR(p...)
    elseif geometry == "T"
        unit = UnitT(p...)
    elseif geometry == "RT"
        unit = UnitRT(p...)
    elseif geometry == "XY"
        unit = UnitXY(p...)
    elseif geometry == "XYT"
        unit = UnitXYT(p...)
    else
        error("Wrong grid geometry.")
    end
    return unit
end


# ******************************************************************************
# R
# ******************************************************************************
struct UnitR{T} <: Unit{T}
    r :: T
    k :: T
    z :: T
    I :: T
end


function UnitR(ru::T, zu::T, Iu::T) where T
    ku = 1 / ru
    return UnitR(ru, ku, zu, Iu)
end


# ******************************************************************************
# T
# ******************************************************************************
struct UnitT{T} <: Unit{T}
    z :: T
    t :: T
    w :: T
    lam :: T
    I :: T
    rho :: T
end


function UnitT(zu::T, tu::T, Iu::T, rhou::T) where T
    wu = 1 / tu
    lamu = tu
    return UnitT(zu, tu, wu, lamu, Iu, rhou)
end


# ******************************************************************************
# RT
# ******************************************************************************
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


function UnitRT(ru::T, zu::T, tu::T, Iu::T, rhou::T) where T
    ku = 1 / ru
    wu = 1 / tu
    lamu = tu
    return UnitRT(ru, ku, zu, tu, wu, lamu, Iu, rhou)
end


# ******************************************************************************
# XY
# ******************************************************************************
struct UnitXY{T} <: Unit{T}
    x :: T
    y :: T
    kx :: T
    ky :: T
    z :: T
    I :: T
end


function UnitXY(xu::T, yu::T, zu::T, Iu::T) where T
    kxu = 1 / xu
    kyu = 1 / yu
    return UnitXY(xu, yu, kxu, kyu, zu, Iu)
end


# ******************************************************************************
# XYT
# ******************************************************************************
struct UnitXYT{T} <: Unit{T}
    x :: T
    y :: T
    kx :: T
    ky :: T
    z :: T
    t :: T
    w :: T
    lam :: T
    I :: T
    rho :: T
end


function UnitXYT(xu::T, yu::T, zu::T, tu::T, Iu::T, rhou::T) where T
    kxu = 1 / xu
    kyu = 1 / yu
    wu = 1 / tu
    lamu = tu
    return UnitXYT(xu, yu, kxu, kyu, zu, tu, wu, lamu, Iu, rhou)
end


# ******************************************************************************
"""
Units of electric field (depends on refractive index n0)
"""
function E(unit::Unit{T}, n0::T) where T
    ksi = n0 * EPS0 * C0 / 2
    Eu = sqrt(unit.I / ksi)
    return convert(T, Eu)
end


end
