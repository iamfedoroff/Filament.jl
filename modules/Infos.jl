module Infos

import Formatting
import Dates

import PyCall

import Fields
import Media
import PlasmaComponents

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum


struct Info
    fname :: String
end


function Info(fname, file_input, file_initial_condition, file_medium,
              unit, grid, field, medium, plasma)
    fmt(x) = Formatting.fmt("18.12e", x)

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

    file_input_content = read(file_input, String)
    file_initial_condition_content = read(file_initial_condition, String)
    file_medium_content = read(file_medium, String)

    a0 = Fields.beam_radius(grid, field) * unit.r
    t0 = Fields.pulse_duration(grid, field) * unit.t
    I0 = Fields.peak_intensity(field) * unit.I
    P = Fields.peak_power(grid, field) * unit.r^2 * unit.I
    Pcr = Media.critical_power(medium, field.w0)

    comp_tab = ""
    for i=1:plasma.Ncomp
        comp = plasma.components[i]
        comp_tab = comp_tab * "$(Formatting.fmt("<25", comp.name)) $(comp.K)\n"
    end

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
dr = $(fmt(grid.dr_mean * unit.r)) [m] - average spatial step
dk = $(fmt(grid.dk_mean * unit.k)) [1/m] - average spatial frequency (angular) step
kc = $(fmt(grid.kc * unit.k)) [1/m] - spatial Nyquist frequency (angular)
dt = $(fmt(grid.dt * unit.t)) [s] - temporal step
df = $(fmt(grid.df * unit.w)) [1/s] - temporal frequency step
fc = $(fmt(grid.fc * unit.w)) [1/s] - temporal Nyquist frequency

# Field ------------------------------------------------------------------------
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

# Medium (all values are given at central frequency) ---------------------------
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

--------------------------------------------------------------------------------
Component's name          multiphoton order K
--------------------------------------------------------------------------------
$comp_tab
********************************************************************************
                              Runtime information
********************************************************************************
"""

    fp = open(fname, "w")
    write(fp, sdata)
    close(fp)

    return Info(fname)
end


function write_message(info, message)
    fp = open(info.fname, "a")
    println(fp, message)
    close(fp)
    println(message)
end


end
