module Equations

import StaticArrays
import CUDAnative


struct PFunction{F<:Function, T<:Tuple} <: Function
    func :: F
    p :: T
end


function (pfunc::PFunction)(x...)
    pfunc.func(x..., pfunc.p)
end


struct Problem{T<:Union{AbstractFloat,AbstractArray}, F<:Function}
    u0 :: T
    step :: F
end


function Problem(alg::String, u0::Union{AbstractFloat,StaticArrays.SVector}, func::Function)
    @assert alg in ("RK2", "RK3", "RK4", "Tsit5", "ATsit5")
    if alg == "RK2"
        step = step_rk2
    elseif alg == "RK3"
        step = step_rk3
    elseif alg == "RK4"
        step = step_rk4
    elseif alg == "Tsit5"
        step = step_tsit5
    elseif alg == "ATsit5"
        step = step_atsit5
    end
    p = (func, )
    pstep = PFunction(step, p)
    return Problem(u0, pstep)
end


function Problem(alg::String, u0::AbstractArray, func::Function)
    @assert alg in ("RK2", "RK3", "RK4", "Tsit5", "ATsit5")
    utmp = similar(u0)
    if alg == "RK2"
        step = step_rk2!
        k1 = similar(u0)
        k2 = similar(u0)
        p = (func, k1, k2, utmp)
    elseif alg == "RK3"
        step = step_rk3!
        k1 = similar(u0)
        k2 = similar(u0)
        k3 = similar(u0)
        p = (func, k1, k2, k3, utmp)
    elseif alg == "RK4"
        step = step_rk4!
        k1 = similar(u0)
        k2 = similar(u0)
        k3 = similar(u0)
        k4 = similar(u0)
        p = (func, k1, k2, k3, k4, utmp)
    elseif alg == "Tsit5"
        step = step_tsit5!
        k1 = similar(u0)
        k2 = similar(u0)
        k3 = similar(u0)
        k4 = similar(u0)
        k5 = similar(u0)
        k6 = similar(u0)
        p = (func, k1, k2, k3, k4, k5, k6, utmp)
    elseif alg == "ATsit5"
        step = step_atsit5!
        k1 = similar(u0)
        k2 = similar(u0)
        k3 = similar(u0)
        k4 = similar(u0)
        k5 = similar(u0)
        k6 = similar(u0)
        uhat = similar(u0)
        etmp = similar(real(u0))   # real valued array for error estimation
        p = (func, k1, k2, k3, k4, k5, k6, utmp, uhat, etmp)
    end
    pstep = PFunction(step, p)
    return Problem(u0, pstep)
end


# ******************************************************************************
# RK2
# ******************************************************************************
function tableau_rk2(T::Type)
    cs = (2. / 3., )   # c2
    as = (2. / 3., )   # a21
    bs = (1. / 4., 3. / 4.)   # b1, b2
    cs = @. convert(T, cs)
    as = @. convert(T, as)
    bs = @. convert(T, bs)
    return cs, as, bs
end


function step_rk2(u::Union{AbstractFloat,StaticArrays.SVector}, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func, = p

    cs, as, bs = tableau_rk2(T)
    a21, = as
    b1, b2 = bs
    c2, = cs

    k1 = func(u, t, args)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, ttmp, args)

    unew = u + dt * (b1 * k1 + b2 * k2)
    tnew = t + dt
    return (unew, tnew)
end


function step_rk2!(u::AbstractArray, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func!, k1, k2, utmp = p

    cs, as, bs = tableau_rk2(T)
    a21, = as
    b1, b2 = bs
    c2, = cs

    func!(k1, u, t, args)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func!(k2, utmp, ttmp, args)

    @. u = u + dt * (b1 * k1 + b2 * k2)
    tnew = t + dt
    return tnew
end


# ******************************************************************************
# RK3
# ******************************************************************************
function tableau_rk3(T::Type)
    cs = (0.5, 1.)   # c2, c3
    as = (0.5, -1., 2.)   # a21, a31, a32
    bs = (1. / 6., 2. / 3., 1. / 6.)   # b1, b2, b3
    cs = @. convert(T, cs)
    as = @. convert(T, as)
    bs = @. convert(T, bs)
    return cs, as, bs
end


function step_rk3(u::Union{AbstractFloat,StaticArrays.SVector}, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func, = p

    cs, as, bs = tableau_rk3(T)
    a21, a31, a32 = as
    b1, b2, b3 = bs
    c2, c3 = cs

    k1 = func(u, t, args)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, ttmp, args)

    utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    k3 = func(utmp, ttmp, args)

    unew = u + dt * (b1 * k1 + b2 * k2 + b3 * k3)
    tnew = t + dt
    return (unew, tnew)
end


function step_rk3!(u::AbstractArray, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func!, k1, k2, k3, utmp = p

    cs, as, bs = tableau_rk3(T)
    a21, a31, a32 = as
    b1, b2, b3 = bs
    c2, c3 = cs

    func!(k1, u, t, args)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func!(k2, utmp, ttmp, args)

    @. utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    func!(k3, utmp, ttmp, args)

    @. u = u + dt * (b1 * k1 + b2 * k2 + b3 * k3)
    tnew = t + dt
    return tnew
end


# ******************************************************************************
# RK4
# ******************************************************************************
function tableau_rk4(T::Type)
    cs = (0.5, 0.5, 1.)   # c2, c3, c4
    as = (0.5, 0., 0.5, 0., 0., 1.)   # a21, a31, a32, a41, a42, a43
    bs = (1. / 6., 1. / 3., 1. / 3., 1. / 6.)   # b1, b2, b3, b4
    cs = @. convert(T, cs)
    as = @. convert(T, as)
    bs = @. convert(T, bs)
    return cs, as, bs
end


function step_rk4(u::Union{AbstractFloat,StaticArrays.SVector}, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func, = p

    cs, as, bs = tableau_rk4(T)
    c2, c3, c4 = cs
    a21, a31, a32, a41, a42, a43 = as
    b1, b2, b3, b4 = bs

    k1 = func(u, t, args)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, ttmp, args)

    utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    k3 = func(utmp, ttmp, args)

    utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    ttmp = t + c4 * dt
    k4 = func(utmp, ttmp, args)

    unew = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
    tnew = t + dt
    return (unew, tnew)
end


function step_rk4!(u::AbstractArray, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func!, k1, k2, k3, k4, utmp = p

    cs, as, bs = tableau_rk4(T)
    c2, c3, c4 = cs
    a21, a31, a32, a41, a42, a43 = as
    b1, b2, b3, b4 = bs

    func!(k1, u, t, args)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func!(k2, utmp, ttmp, args)

    @. utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    func!(k3, utmp, ttmp, args)

    @. utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    ttmp = t + c4 * dt
    func!(k4, utmp, ttmp, args)

    @. u = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
    tnew = t + dt
    return tnew
end


# ******************************************************************************
# Tsit5
# ******************************************************************************
function tableau_tsit5(T::Type)
    cs = (0.161, 0.327, 0.9, 0.9800255409045097, 1.)   # c2, c3, c4, c5, c6
    as = (0.161,   # a21
          -0.008480655492356989,   # a31
          0.335480655492357,   # a32
          2.8971530571054935,   # a41
          -6.359448489975075,   # a42
          4.3622954328695815,   # a43
          5.325864828439257,   # a51
          -11.748883564062828,   # a52
          7.4955393428898365,   # a53
          -0.09249506636175525,   # a54
          5.86145544294642,   # a61
          -12.92096931784711,   # a62
          8.159367898576159,   # a63
          -0.071584973281401,   # a64
          -0.028269050394068383)   # a65
    bs = (0.09646076681806523,   # b1
          0.01,   # b2
          0.4798896504144996,   # b3
          1.379008574103742,   # b4
          -3.290069515436081,   # b5
          2.324710524099774)   # b6
    cs = @. convert(T, cs)
    as = @. convert(T, as)
    bs = @. convert(T, bs)
    return cs, as, bs
end


function step_tsit5(u::Union{AbstractFloat,StaticArrays.SVector}, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func, = p

    cs, as, bs = tableau_tsit5(T)
    c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs

    k1 = func(u, t, args)

    utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    k2 = func(utmp, ttmp, args)

    utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    k3 = func(utmp, ttmp, args)

    utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    ttmp = t + c4 * dt
    k4 = func(utmp, ttmp, args)

    utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    ttmp = t + c5 * dt
    k5 = func(utmp, ttmp, args)

    utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    ttmp = t + c6 * dt
    k6 = func(utmp, ttmp, args)

    unew = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    tnew = t + dt
    return (unew, tnew)
end


function step_tsit5!(u::AbstractArray, t::T, dt::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func!, k1, k2, k3, k4, k5, k6, utmp = p

    cs, as, bs = tableau_tsit5(T)
    c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs

    func!(k1, u, t, args)

    @. utmp = u + dt * a21 * k1
    ttmp = t + c2 * dt
    func!(k2, utmp, ttmp, args)

    @. utmp = u + dt * (a31 * k1 + a32 * k2)
    ttmp = t + c3 * dt
    func!(k3, utmp, ttmp, args)

    @. utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    ttmp = t + c4 * dt
    func!(k4, utmp, ttmp, args)

    @. utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    ttmp = t + c5 * dt
    func!(k5, utmp, ttmp, args)

    @. utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    ttmp = t + c6 * dt
    func!(k6, utmp, ttmp, args)

    @. u = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    tnew = t + dt
    return tnew
end


# ******************************************************************************
# ATsit5
# ******************************************************************************
function tableau_atsit5(T::Type)
    cs, as, bs = tableau_tsit5(T)
    bhats = (0.00178001105222577714,   # bhat1
             0.0008164344596567469,   # bhat2
             -0.007880878010261995,   # bhat3
             0.1447110071732629,   # bhat4
             -0.5823571654525552,   # bhat5
             0.45808210592918697)   # bhat6
    bhats = @. convert(T, bhats)
    return cs, as, bs, bhats
end


function step_atsit5(u::Union{AbstractFloat,StaticArrays.SVector}, t::T, dt0::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func, = p

    cs, as, bs, bhats = tableau_atsit5(T)
    c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = bhats

    err = Inf
    dt = dt0
    unew = zero(u)

    while err > 1
        k1 = func(u, t, args)

        utmp = u + dt * a21 * k1
        ttmp = t + c2 * dt
        k2 = func(utmp, ttmp, args)

        utmp = u + dt * (a31 * k1 + a32 * k2)
        ttmp = t + c3 * dt
        k3 = func(utmp, ttmp, args)

        utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        ttmp = t + c4 * dt
        k4 = func(utmp, ttmp, args)

        utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        ttmp = t + c5 * dt
        k5 = func(utmp, ttmp, args)

        utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        ttmp = t + c6 * dt
        k6 = func(utmp, ttmp, args)

        unew = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

        # Error estimation:
        atol = convert(T, 1e-3)   # absolute tolerance
        rtol = convert(T, 1e-3)   # relative tolerance

        uhat = u + dt * (bhat1 * k1 + bhat2 * k2 + bhat3 * k3 + bhat4 * k4 +
                         bhat5 * k5 + bhat6 * k6)

        etmp = @. atol + rtol * max(abs(u), abs(utmp))
        etmp = @. abs(utmp - uhat) / etmp
        # etmp = @. etmp^2
        # err = sqrt(sum(etmp) / length(etmp))
        # if err > 1
        #     dt = convert(T, 0.9) * dt / err^convert(T, 0.2)   # 0.2 = 1/5
        # end
        etmp = @. CUDAnative.pow(etmp, 2)
        err = CUDAnative.sqrt(CUDAnative.sum(etmp) / length(etmp))
        if err > 1
            dt = convert(T, 0.9) * dt / CUDAnative.pow(err, convert(T, 0.2))   # 0.2 = 1/5
        end
    end

    tnew = t + dt
    return (unew, tnew)
end


function step_atsit5!(u::AbstractArray, t::T, dt0::T, args::Tuple, p::Tuple) where T<:AbstractFloat
    func!, k1, k2, k3, k4, k5, k6, utmp, uhat, etmp = p

    cs, as, bs, bhats = tableau_atsit5(T)
    c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = as
    b1, b2, b3, b4, b5, b6 = bs
    bhat1, bhat2, bhat3, bhat4, bhat5, bhat6 = bhats

    err = Inf
    dt = dt0

    while err > 1
        func!(k1, u, t, args)

        @. utmp = u + dt * a21 * k1
        ttmp = t + c2 * dt
        func!(k2, utmp, ttmp, args)

        @. utmp = u + dt * (a31 * k1 + a32 * k2)
        ttmp = t + c3 * dt
        func!(k3, utmp, ttmp, args)

        @. utmp = u + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        ttmp = t + c4 * dt
        func!(k4, utmp, ttmp, args)

        @. utmp = u + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        ttmp = t + c5 * dt
        func!(k5, utmp, ttmp, args)

        @. utmp = u + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        ttmp = t + c6 * dt
        func!(k6, utmp, ttmp, args)

        @. utmp = u + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

        # Error estimation:
        atol = convert(T, 1e-3)   # absolute tolerance
        rtol = convert(T, 1e-3)   # relative tolerance

        @. uhat = u + dt * (bhat1 * k1 + bhat2 * k2 + bhat3 * k3 + bhat4 * k4 +
                            bhat5 * k5 + bhat6 * k6)

        @. etmp = atol + rtol * max(abs(u), abs(utmp))
        @. etmp = (abs(utmp - uhat) / etmp)^2
        err = sqrt(sum(etmp) / length(etmp))
        if err > 1
            dt = convert(T, 0.9) * dt / err^convert(T, 0.2)   # 0.2 = 1/5
            # dt = convert(T, 0.9) * dt / CUDAnative.pow(err, convert(T, 0.2))   # 0.2 = 1/5
        end
    end

    @. u = utmp
    tnew = t + dt
    return tnew
end


"""
Complex version of CUDAnative.abs function.
"""
@inline function CUDAnative.abs(x::Complex{T}) where T
    return CUDAnative.sqrt(x.re * x.re + x.im * x.im)
end


end
