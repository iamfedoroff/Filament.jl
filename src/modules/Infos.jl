module Infos

import Dates
import Formatting

import ..Constants: C0, HBAR
import ..FieldAnalyzers
import ..Fields
import ..Grids
import ..Media
import ..Units


fmt(x) = Formatting.fmt("18.12e", x)


struct Info
    fname :: String
end


function Info(
    fname::String,
    unit::Units.Unit,
    grid::Grids.Grid,
    field::Fields.Field,
    medium::Media.Medium,
    analyzer::FieldAnalyzers.FieldAnalyzer,
    file_input::String,
    file_initial_condition::String,
    file_medium::String,
)
    revision = vcs_revision()

    file_input_content = read(file_input, String)
    file_initial_condition_content = read(file_initial_condition, String)
    file_medium_content = read(file_medium, String)

    sdata_grid = info_grid(unit, grid)
    sdata_field = info_field(unit, grid, field, analyzer)
    sdata_medium = info_medium(unit, grid, field, medium, analyzer)

    sdata =
"""
********************************************************************************
                                   jlFilament
********************************************************************************
datetime: $(Dates.now())
revision: $revision

********************************************************************************
                                   Input file
********************************************************************************
$file_input_content
********************************************************************************
                             Initial condition file
********************************************************************************
$file_initial_condition_content
********************************************************************************
                                  Medium file
********************************************************************************
$file_medium_content
********************************************************************************
                                   Parameters
********************************************************************************
# Grid -------------------------------------------------------------------------
$sdata_grid
# Field ------------------------------------------------------------------------
$sdata_field
# Medium (all values are given at central frequency) ---------------------------
$sdata_medium
********************************************************************************
                              Runtime information
********************************************************************************
"""

    fp = open(fname, "w")
    write(fp, sdata)
    close(fp)

    return Info(fname)
end


function vcs_revision()
    revision = "unavailable"
    cwdir = pwd()
    try
        cd(@__DIR__)
        branch = read(@cmd("git rev-parse --abbrev-ref HEAD"), String)
        hash = read(@cmd("git rev-parse HEAD"), String)
        revision = string(strip(branch) * " " * strip(hash))
    catch
        @warn "The VCS revision is unavailable."
    end
    cd(cwdir)
    return revision
end


function info_grid(unit::Units.UnitR, grid::Grids.GridR)
    dr_mean = sum(diff(grid.r)) / length(diff(grid.r))   # spatial step
    dk_mean = sum(diff(grid.k)) / length(diff(grid.k))   # spatial frequency step
    kc = 2 * pi * 0.5 / dr_mean   # spatial Nyquist frequency

    sdata =
    """
    dr = $(fmt(dr_mean * unit.r)) [m] - average spatial step
    dk = $(fmt(dk_mean * unit.k)) [1/m] - average spatial frequency (angular) step
    kc = $(fmt(kc * unit.k)) [1/m] - spatial Nyquist frequency (angular)
    """
    return sdata
end


function info_grid(unit::Units.UnitT, grid::Grids.GridT)
    f = grid.w / (2 * pi)
    df = f[2] - f[1]
    fc = 0.5 / grid.dt   # temporal Nyquist frequency

    sdata =
    """
    dt = $(fmt(grid.dt * unit.t)) [s] - temporal step
    df = $(fmt(df * unit.w)) [1/s] - temporal frequency step
    fc = $(fmt(fc * unit.w)) [1/s] - temporal Nyquist frequency
    """
    return sdata
end


function info_grid(unit::Units.UnitRT, grid::Grids.GridRT)
    dr_mean = sum(diff(grid.r)) / length(diff(grid.r))   # average spatial step
    dk_mean = sum(diff(grid.k)) / length(diff(grid.k))   # spatial frequency step
    kc = 2 * pi * 0.5 / dr_mean   # spatial Nyquist frequency
    f = grid.w / (2 * pi)
    df = f[2] - f[1]
    fc = 0.5 / grid.dt   # temporal Nyquist frequency

    sdata =
    """
    dr = $(fmt(dr_mean * unit.r)) [m] - average spatial step
    dk = $(fmt(dk_mean * unit.k)) [1/m] - average spatial frequency (angular) step
    kc = $(fmt(kc * unit.k)) [1/m] - spatial Nyquist frequency (angular)
    dt = $(fmt(grid.dt * unit.t)) [s] - temporal step
    df = $(fmt(df * unit.w)) [1/s] - temporal frequency step
    fc = $(fmt(fc * unit.w)) [1/s] - temporal Nyquist frequency
    """
    return sdata
end


function info_grid(unit::Units.UnitXY, grid::Grids.GridXY)
    dkx = grid.kx[2] - grid.kx[1]
    dky = grid.ky[2] - grid.ky[1]
    kxc = 2 * pi * 0.5 / dkx   # x angular spatial Nyquist frequency
    kyc = 2 * pi * 0.5 / dky   # y angular spatial Nyquist frequency

    sdata =
    """
    dx  = $(fmt(grid.dx * unit.x)) [m] - x spatial step
    dy  = $(fmt(grid.dy * unit.y)) [m] - y spatial step
    dkx = $(fmt(dkx * unit.kx)) [1/m] - x spatial frequency (angular) step
    dky = $(fmt(dky * unit.ky)) [1/m] - y spatial frequency (angular) step
    kxc  = $(fmt(kxc * unit.kx)) [1/m] - x spatial Nyquist frequency (angular)
    kyc  = $(fmt(kyc * unit.ky)) [1/m] - y spatial Nyquist frequency (angular)
    """
    return sdata
end


function info_grid(unit::Units.UnitXYT, grid::Grids.GridXYT)
    dkx = grid.kx[2] - grid.kx[1]
    dky = grid.ky[2] - grid.ky[1]
    kxc = 2 * pi * 0.5 / dkx   # x angular spatial Nyquist frequency
    kyc = 2 * pi * 0.5 / dky   # y angular spatial Nyquist frequency
    f = grid.w / (2 * pi)
    df = f[2] - f[1]
    fc = 0.5 / grid.dt   # temporal Nyquist frequency

    sdata =
    """
    dx  = $(fmt(grid.dx * unit.x)) [m] - x spatial step
    dy  = $(fmt(grid.dy * unit.y)) [m] - y spatial step
    dkx = $(fmt(dkx * unit.kx)) [1/m] - x spatial frequency (angular) step
    dky = $(fmt(dky * unit.ky)) [1/m] - y spatial frequency (angular) step
    kxc  = $(fmt(kxc * unit.kx)) [1/m] - x spatial Nyquist frequency (angular)
    kyc  = $(fmt(kyc * unit.ky)) [1/m] - y spatial Nyquist frequency (angular)
    dt = $(fmt(grid.dt * unit.t)) [s] - temporal step
    df = $(fmt(df * unit.w)) [1/s] - temporal frequency step
    fc = $(fmt(fc * unit.w)) [1/s] - temporal Nyquist frequency
    """
    return sdata
end


function info_field(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.Field,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0
    f0 = w0 / (2 * pi)
    lam0 = 2 * pi * C0 / w0

    a0 = analyzer.rfil * unit.r
    a0e2 = sqrt(2) * a0
    a0fwhm = 2 * sqrt(log(2)) * a0
    I0 = analyzer.Imax * unit.I
    P = analyzer.P * unit.r^2 * unit.I

    Pg = pi * a0^2 * I0
    Wph = HBAR * w0

    sdata =
    """
    a0e    = $(fmt(a0)) [m] - initial beam size (half width 1/e)
    a0e2   = $(fmt(a0e2)) [m] - initial beam size (half width 1/e^2)
    a0fwhm = $(fmt(a0fwhm)) [m] - initial beam size (FWHM)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(lam0)) [m] - central wavelength
    f0   = $(fmt(f0)) [1/s] - central frequency
    w0   = $(fmt(w0)) [1/s] - central frequency (angular)
    P    = $(fmt(P)) [W] - peak power
    Pg   = $(fmt(Pg)) [W] - Gaussian peak power (pi*a0^2*I0)
    Wph  = $(fmt(Wph)) [J] - energy of one photon
    """
    return sdata
end


function info_field(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.Field,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0
    f0 = field.w0 / (2 * pi)
    lam0 = 2 * pi * C0 / w0

    t0 = analyzer.duration * unit.t
    t0e2 = sqrt(2) * t0
    t0fwhm = 2 * sqrt(log(2)) * t0
    I0 = analyzer.Imax * unit.I
    F = analyzer.F * unit.t * unit.I

    Fg = sqrt(pi) * t0 * I0
    Wph = HBAR * w0

    sdata =
    """
    t0e     = $(fmt(t0)) [s] - initial pulse duration (half width 1/e)
    t0e2    = $(fmt(t0e2)) [s] - initial pulse duration (half width 1/e^2)
    t0fwhm  = $(fmt(t0fwhm)) [s] - initial pulse duration (FWHM)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(lam0)) [m] - central wavelength
    f0   = $(fmt(f0)) [1/s] - central frequency
    w0   = $(fmt(w0)) [1/s] - central frequency (angular)
    F    = $(fmt(F)) [J/m^2] - peak fluence
    Fg   = $(fmt(Fg)) [J/m^2] - Gaussian peak fluence (pi^0.5*t0*I0)
    Wph  = $(fmt(Wph)) [J] - energy of one photon
    """
    return sdata
end


function info_field(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.Field,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0
    f0 = field.w0 / (2 * pi)
    lam0 = 2 * pi * C0 / w0

    a0 = analyzer.rfil * unit.r
    a0e2 = sqrt(2) * a0
    a0fwhm = 2 * sqrt(log(2)) * a0
    t0 = analyzer.tau * unit.t
    t0e2 = sqrt(2) * t0
    t0fwhm = 2 * sqrt(log(2)) * t0
    I0 = analyzer.Imax * unit.I
    F = analyzer.Fmax * unit.t * unit.I
    W = analyzer.W * unit.r^2 * unit.t * unit.I

    P = W / t0 / sqrt(pi)
    Pg = pi * a0^2 * I0
    Fg = sqrt(pi) * t0 * I0
    Wg = pi^1.5 * t0 * a0^2 * I0
    Wph = HBAR * w0

    sdata =
    """
    a0e     = $(fmt(a0)) [m] - initial beam size (half width 1/e)
    a0e2    = $(fmt(a0e2)) [m] - initial beam size (half width 1/e^2)
    a0fwhm  = $(fmt(a0fwhm)) [m] - initial beam size (FWHM)
    t0e     = $(fmt(t0)) [s] - initial pulse duration (half width 1/e)
    t0e2    = $(fmt(t0e2)) [s] - initial pulse duration (half width 1/e^2)
    t0fwhm  = $(fmt(t0fwhm)) [s] - initial pulse duration (FWHM)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(lam0)) [m] - central wavelength
    f0   = $(fmt(f0)) [1/s] - central frequency
    w0   = $(fmt(w0)) [1/s] - central frequency (angular)
    P    = $(fmt(P)) [W] - peak power
    Pg   = $(fmt(Pg)) [W] - Gaussian peak power (pi*a0^2*I0)
    F    = $(fmt(F)) [J/m^2] - peak fluence
    Fg   = $(fmt(Fg)) [J/m^2] - Gaussian peak fluence (pi^0.5*t0*I0)
    W    = $(fmt(W)) [J] - pulse energy
    Wg   = $(fmt(Wg)) [J] - Gaussian pulse energy (pi^1.5*t0*a0^2*I0)
    Wph  = $(fmt(Wph)) [J] - energy of one photon
    """
    return sdata
end


function info_field(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.Field,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0
    f0 = w0 / (2 * pi)
    lam0 = 2 * pi * C0 / w0

    a0 = sqrt(analyzer.ax * analyzer.ay * unit.x * unit.y)
    a0e2 = sqrt(2) * a0
    a0fwhm = 2 * sqrt(log(2)) * a0
    I0 = analyzer.Imax * unit.I
    P = analyzer.P * unit.x * unit.y * unit.I

    Pg = pi * a0^2 * I0
    Wph = HBAR * w0

    sdata =
    """
    a0e    = $(fmt(a0)) [m] - initial beam size (half width 1/e)
    a0e2   = $(fmt(a0e2)) [m] - initial beam size (half width 1/e^2)
    a0fwhm = $(fmt(a0fwhm)) [m] - initial beam size (FWHM)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(lam0)) [m] - central wavelength
    f0   = $(fmt(f0)) [1/s] - central frequency
    w0   = $(fmt(w0)) [1/s] - central frequency (angular)
    P    = $(fmt(P)) [W] - peak power
    Pg   = $(fmt(Pg)) [W] - Gaussian peak power (pi*a0^2*I0)
    Wph  = $(fmt(Wph)) [J] - energy of one photon
    """
    return sdata
end


function info_field(
    unit::Units.UnitXYT,
    grid::Grids.GridXYT,
    field::Fields.Field,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0
    f0 = field.w0 / (2 * pi)
    lam0 = 2 * pi * C0 / w0

    I0 = analyzer.Imax * unit.I
    Wph = HBAR * w0

    sdata =
    """
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(lam0)) [m] - central wavelength
    f0   = $(fmt(f0)) [1/s] - central frequency
    w0   = $(fmt(w0)) [1/s] - central frequency (angular)
    Wph  = $(fmt(Wph)) [J] - energy of one photon
    """
    return sdata
end


function info_medium(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.Field,
    medium::Media.Medium,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0

    a0 = analyzer.rfil * unit.r
    I0 = analyzer.Imax * unit.I
    P = analyzer.P * unit.r^2 * unit.I

    Pcr = Media.critical_power(medium, w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, w0))) [m/V] - 3rd order nonlinear susceptibility
    P   = $(fmt(P / Pcr)) [Pcr] - peak power
    Pcr = $(fmt(Pcr)) [W] - critical power
    Ld     = $(fmt(Media.diffraction_length(medium, w0, a0))) [m] - diffraction length
    La     = $(fmt(Media.absorption_length(medium, w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, w0, I0))) [m] - length of Kerr nonlinearity
    zf     = $(fmt(Media.selffocusing_length(medium, w0, a0, P))) [m] - self-focusing distance
    """
    return sdata
end


function info_medium(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.Field,
    medium::Media.Medium,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0

    t0 = analyzer.duration * unit.t
    I0 = analyzer.Imax * unit.I

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, w0))) [m/V] - 3rd order nonlinear susceptibility
    Ldisp  = $(fmt(Media.dispersion_length(medium, w0, t0))) [m] - dispersion length
    Ldisp3 = $(fmt(Media.dispersion_length3(medium, w0, t0))) [m] - 3rd order dispersion length (t0^3/k3)
    La     = $(fmt(Media.absorption_length(medium, w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, w0, I0))) [m] - length of Kerr nonlinearity
    """
    return sdata
end


function info_medium(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.Field,
    medium::Media.Medium,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0

    a0 = analyzer.rfil * unit.r
    t0 = analyzer.tau * unit.t
    I0 = analyzer.Imax * unit.I
    W = analyzer.W * unit.r^2 * unit.t * unit.I

    P = W / t0 / sqrt(pi)
    Pcr = Media.critical_power(medium, w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, w0))) [m/V] - 3rd order nonlinear susceptibility
    P   = $(fmt(P / Pcr)) [Pcr] - peak power
    Pcr = $(fmt(Pcr)) [W] - critical power
    Ld     = $(fmt(Media.diffraction_length(medium, w0, a0))) [m] - diffraction length
    Ldisp  = $(fmt(Media.dispersion_length(medium, w0, t0))) [m] - dispersion length
    Ldisp3 = $(fmt(Media.dispersion_length3(medium, w0, t0))) [m] - 3rd order dispersion length (t0^3/k3)
    La     = $(fmt(Media.absorption_length(medium, w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, w0, I0))) [m] - length of Kerr nonlinearity
    zf     = $(fmt(Media.selffocusing_length(medium, w0, a0, P))) [m] - self-focusing distance
    """
    return sdata
end


function info_medium(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.Field,
    medium::Media.Medium,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0

    a0 = sqrt(analyzer.ax * analyzer.ay * unit.x * unit.y)
    I0 = analyzer.Imax * unit.I
    P = analyzer.P * unit.x * unit.y * unit.I

    Pcr = Media.critical_power(medium, w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, w0))) [m/V] - 3rd order nonlinear susceptibility
    P   = $(fmt(P / Pcr)) [Pcr] - peak power
    Pcr = $(fmt(Pcr)) [W] - critical power
    Ld     = $(fmt(Media.diffraction_length(medium, w0, a0))) [m] - diffraction length
    La     = $(fmt(Media.absorption_length(medium, w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, w0, I0))) [m] - length of Kerr nonlinearity
    zf     = $(fmt(Media.selffocusing_length(medium, w0, a0, P))) [m] - self-focusing distance
    """
    return sdata
end


function info_medium(
    unit::Units.UnitXYT,
    grid::Grids.GridXYT,
    field::Fields.Field,
    medium::Media.Medium,
    analyzer::FieldAnalyzers.FieldAnalyzer,
)
    w0 = field.w0

    I0 = analyzer.Imax * unit.I
    Pcr = Media.critical_power(medium, w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, w0))) [m/V] - 3rd order nonlinear susceptibility
    Pcr = $(fmt(Pcr)) [W] - critical power
    La     = $(fmt(Media.absorption_length(medium, w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, w0, I0))) [m] - length of Kerr nonlinearity
    """
    return sdata
end


function info_plasma(components)
    sdata =
    """
    --------------------------------------------------------------------------------
    Component's name          multiphoton order K
    --------------------------------------------------------------------------------
    """
    for comp in components
        sdata = sdata * "$(Formatting.fmt("<25", comp.name)) $(comp.K)\n"
    end
    return sdata
end


function write_message(info, message)
    fp = open(info.fname, "a")
    println(fp, message)
    close(fp)
    println(message)
end


end
