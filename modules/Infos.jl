module Infos

import Formatting
import Dates

import Units
import Grids
import Fields
import Media

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum

fmt(x) = Formatting.fmt("18.12e", x)


struct Info
    fname :: String
end


function Info(fname::String, file_input::String, file_initial_condition::String,
              file_medium::String, unit::Units.Unit, grid::Grids.Grid,
              field::Fields.Field, medium::Media.Medium)
    revision = vcs_revision()

    file_input_content = read(file_input, String)
    file_initial_condition_content = read(file_initial_condition, String)
    file_medium_content = read(file_medium, String)

    sdata_grid = info_grid(unit, grid)
    sdata_field = info_field(unit, grid, field)
    sdata_medium = info_medium(unit, grid, field, medium)

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
        hg_num = read(@cmd("hg id -n"), String)
        hg_id = read(@cmd("hg id"), String)
        revision = string(strip(hg_num), ":", strip(hg_id))
    catch
    end
    cd(cwdir)
    return revision
end


function info_grid(unit::Units.UnitR, grid::Grids.GridR)
    sdata =
    """
    dr = $(fmt(grid.dr_mean * unit.r)) [m] - average spatial step
    dk = $(fmt(grid.dk_mean * unit.k)) [1/m] - average spatial frequency (angular) step
    kc = $(fmt(grid.kc * unit.k)) [1/m] - spatial Nyquist frequency (angular)
    """
    return sdata
end


function info_grid(unit::Units.UnitRT, grid::Grids.GridRT)
    sdata =
    """
    dr = $(fmt(grid.dr_mean * unit.r)) [m] - average spatial step
    dk = $(fmt(grid.dk_mean * unit.k)) [1/m] - average spatial frequency (angular) step
    kc = $(fmt(grid.kc * unit.k)) [1/m] - spatial Nyquist frequency (angular)
    dt = $(fmt(grid.dt * unit.t)) [s] - temporal step
    df = $(fmt(grid.df * unit.w)) [1/s] - temporal frequency step
    fc = $(fmt(grid.fc * unit.w)) [1/s] - temporal Nyquist frequency
    """
    return sdata
end


function info_grid(unit::Units.UnitXY, grid::Grids.GridXY)
    sdata =
    """
    dx  = $(fmt(grid.dx * unit.x)) [m] - x spatial step
    dy  = $(fmt(grid.dy * unit.y)) [m] - y spatial step
    dkx = $(fmt(grid.dkx * unit.kx)) [1/m] - x spatial frequency (angular) step
    dky = $(fmt(grid.dky * unit.ky)) [1/m] - y spatial frequency (angular) step
    kxc  = $(fmt(grid.kxc * unit.kx)) [1/m] - x spatial Nyquist frequency (angular)
    kyc  = $(fmt(grid.kyc * unit.ky)) [1/m] - y spatial Nyquist frequency (angular)
    """
    return sdata
end


function info_field(unit::Units.UnitR, grid::Grids.GridR,
                    field::Fields.FieldR)
    a0 = Fields.beam_radius(grid, field) * unit.r
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.r^2 * unit.I

    sdata =
    """
    a0   = $(fmt(a0)) [m] - initial beam radius (1/e)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(field.lam0)) [m] - central wavelength
    f0   = $(fmt(field.f0)) [1/s] - central frequency
    w0   = $(fmt(field.w0)) [1/s] - central frequency (angular)
    P    = $(fmt(P)) [W] - peak power
    Pg   = $(fmt(Fields.peak_power_gauss(grid, field) * unit.r^2 * unit.I)) [W] - Gaussian peak power (pi*a0^2*I0)
    Wph  = $(fmt(Fields.energy_photon(field))) [J] - energy of one photon
    """
    return sdata
end


function info_field(unit::Units.UnitRT, grid::Grids.GridRT,
                    field::Fields.FieldRT)
    a0 = Fields.beam_radius(grid, field) * unit.r
    t0 = Fields.pulse_duration(grid, field) * unit.t
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.r^2 * unit.I

    sdata =
    """
    a0   = $(fmt(a0)) [m] - initial beam radius (1/e)
    t0   = $(fmt(t0)) [s] - initial pulse duration (half width 1/e)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(field.lam0)) [m] - central wavelength
    f0   = $(fmt(field.f0)) [1/s] - central frequency
    w0   = $(fmt(field.w0)) [1/s] - central frequency (angular)
    P    = $(fmt(P)) [W] - peak power
    Pg   = $(fmt(Fields.peak_power_gauss(grid, field) * unit.r^2 * unit.I)) [W] - Gaussian peak power (pi*a0^2*I0)
    F    = $(fmt(Fields.peak_fluence(grid, field) * unit.t * unit.I)) [J/m^2] - peak fluence
    Fg   = $(fmt(Fields.peak_fluence_gauss(grid, field) * unit.t * unit.I)) [J/m^2] - Gaussian peak fluence (pi^0.5*t0*I0)
    W    = $(fmt(Fields.energy(grid, field) * unit.r^2 * unit.t * unit.I)) [J] - pulse energy
    Wg   = $(fmt(Fields.energy_gauss(grid, field) * unit.r^2 * unit.t * unit.I)) [J] - Gaussian pulse energy (pi^1.5*t0*a0^2*I0)
    Wph  = $(fmt(Fields.energy_photon(field))) [J] - energy of one photon
    """
    return sdata
end


function info_field(unit::Units.UnitXY, grid::Grids.GridXY,
                    field::Fields.FieldXY)
    a0 = Fields.beam_radius(grid, field) * sqrt(unit.x * unit.y)
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.x * unit.y * unit.I
    Pgauss = Fields.peak_power_gauss(grid, field) * unit.x * unit.y * unit.I
    Wph = Fields.energy_photon(field)

    sdata =
    """
    a0   = $(fmt(a0)) [m] - initial beam radius (1/e)
    I0   = $(fmt(I0)) [W/m^2] - initial intensity
    lam0 = $(fmt(field.lam0)) [m] - central wavelength
    f0   = $(fmt(field.f0)) [1/s] - central frequency
    w0   = $(fmt(field.w0)) [1/s] - central frequency (angular)
    P    = $(fmt(P)) [W] - peak power
    Pg   = $(fmt(Pgauss)) [W] - Gaussian peak power (pi*a0^2*I0)
    Wph  = $(fmt(Wph)) [J] - energy of one photon
    """
    return sdata
end


function info_medium(unit::Units.UnitR, grid::Grids.GridR,
                     field::Fields.FieldR, medium::Media.Medium)
    a0 = Fields.beam_radius(grid, field) * unit.r
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.r^2 * unit.I
    Pcr = Media.critical_power(medium, field.w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, field.w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, field.w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, field.w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, field.w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, field.w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, field.w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, field.w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, field.w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, field.w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, field.w0))) [m/V] - 3rd order nonlinear susceptibility
    P   = $(fmt(P / Pcr)) [Pcr] - peak power
    Pcr = $(fmt(Pcr)) [W] - critical power
    Ld     = $(fmt(Media.diffraction_length(medium, field.w0, a0))) [m] - diffraction length
    La     = $(fmt(Media.absorption_length(medium, field.w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, field.w0, I0))) [m] - length of Kerr nonlinearity
    zf     = $(fmt(Media.selffocusing_length(medium, field.w0, a0, P))) [m] - self-focusing distance
    """
    return sdata
end


function info_medium(unit::Units.UnitRT, grid::Grids.GridRT,
                     field::Fields.FieldRT, medium::Media.Medium)
    a0 = Fields.beam_radius(grid, field) * unit.r
    t0 = Fields.pulse_duration(grid, field) * unit.t
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.r^2 * unit.I
    Pcr = Media.critical_power(medium, field.w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, field.w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, field.w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, field.w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, field.w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, field.w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, field.w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, field.w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, field.w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, field.w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, field.w0))) [m/V] - 3rd order nonlinear susceptibility
    P   = $(fmt(P / Pcr)) [Pcr] - peak power
    Pcr = $(fmt(Pcr)) [W] - critical power
    Ld     = $(fmt(Media.diffraction_length(medium, field.w0, a0))) [m] - diffraction length
    Ldisp  = $(fmt(Media.dispersion_length(medium, field.w0, t0))) [m] - dispersion length
    Ldisp3 = $(fmt(Media.dispersion_length3(medium, field.w0, t0))) [m] - 3rd order dispersion length (t0^3/k3)
    La     = $(fmt(Media.absorption_length(medium, field.w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, field.w0, I0))) [m] - length of Kerr nonlinearity
    zf     = $(fmt(Media.selffocusing_length(medium, field.w0, a0, P))) [m] - self-focusing distance
    """
    return sdata
end


function info_medium(unit::Units.UnitXY, grid::Grids.GridXY,
                     field::Fields.FieldXY, medium::Media.Medium)
    a0 = Fields.beam_radius(grid, field) * sqrt(unit.x * unit.y)
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.x * unit.y * unit.I
    Pcr = Media.critical_power(medium, field.w0)

    sdata =
    """
    n0re = $(fmt(real(Media.refractive_index(medium, field.w0)))) [-] - refractive index (real part)
    n0im = $(fmt(imag(Media.refractive_index(medium, field.w0)))) [-] - refractive index (imaginary part)
    k0 = $(fmt(Media.k_func(medium, field.w0))) [1/m] - wave number
    k1 = $(fmt(Media.k1_func(medium, field.w0))) [s/m] - 1st derivative of wave number: d(k0)/dw
    k2 = $(fmt(Media.k2_func(medium, field.w0))) [s^2/m] - 2nd derivative of wave number: d(k1)/dw
    k3 = $(fmt(Media.k3_func(medium, field.w0))) [s^3/m] - 3rd derivative of wave number: d(k2)/dw
    ga = $(fmt(Media.absorption_coefficient(medium, field.w0))) [1/m] - linear absorption coefficient (by field)
    vp = $(fmt(Media.phase_velocity(medium, field.w0) / C0)) [C0] - phase velocity
    vg = $(fmt(Media.group_velocity(medium, field.w0) / C0)) [C0] - group velocity
    n2  = $(fmt(medium.n2)) [m^2/W] - Kerr nonlinear index
    chi3 = $(fmt(Media.chi3_func(medium, field.w0))) [m/V] - 3rd order nonlinear susceptibility
    P   = $(fmt(P / Pcr)) [Pcr] - peak power
    Pcr = $(fmt(Pcr)) [W] - critical power
    Ld     = $(fmt(Media.diffraction_length(medium, field.w0, a0))) [m] - diffraction length
    La     = $(fmt(Media.absorption_length(medium, field.w0))) [m] - linear absorption length
    Lnl    = $(fmt(Media.nonlinearity_length(medium, field.w0, I0))) [m] - length of Kerr nonlinearity
    zf     = $(fmt(Media.selffocusing_length(medium, field.w0, a0, P))) [m] - self-focusing distance
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
