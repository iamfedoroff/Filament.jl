module Input

# Global packages:
import CUDA
import FFTW
import ODEIntegrators
import StaticArrays

# Local package-like modules:
import ..TabulatedFunctions

# Local modules:
import ..Constants: FloatGPU, C0, EPS0, MU0, QE, ME, HBAR
import ..Grids
import ..Media
import ..Units

const DEFPATHNR = joinpath(@__DIR__, "modules", "medium_responses")
const DEFPATHPE = joinpath(@__DIR__, "modules", "plasma_equations")


function prepare(fname)
    file_input = abspath(fname)
    include(file_input)
    cd(dirname(file_input))

    include(abspath(file_initial_condition))
    include(abspath(file_medium))

    p_medium = (permittivity, permeability, n2)
    p_info = (file_input, abspath(file_initial_condition), abspath(file_medium))

    if NONLINEARITY
        responses_local = responses
    else
        responses_local = []
    end

    if geometry == "R"
        z_local = FloatGPU(z / zu)
        zmax_local = FloatGPU(zmax)
        lam0_local = FloatGPU(lam0)
        dz_plothdf_local = FloatGPU(dz_plothdf)

        p_unit = (ru, zu, Iu)
        p_grid = (FloatGPU(rmax), Nr)
        p_field = (FloatGPU(lam0), initial_condition, HTLOAD, file_ht)
        p_guard = (FloatGPU(rguard), FloatGPU(kguard))
        p_dzadaptive = (dz_initial, dzphimax, NONLINEARITY)

        model_keys = (
            NONLINEARITY=NONLINEARITY,
            PLASMA=false,
            KPARAXIAL=KPARAXIAL,
            QPARAXIAL=QPARAXIAL,
            ALG=ALG,
        )
        p_model = (responses_local, Dict(), model_keys)
    elseif geometry == "T"
        z_local = z / zu
        zmax_local = zmax
        lam0_local = lam0
        dz_plothdf_local = dz_plothdf

        p_unit = (zu, tu, Iu, rhou)
        p_grid = (tmin, tmax, Nt)
        p_field = (lam0, initial_condition)
        p_guard = (tguard, wguard)
        p_dzadaptive = (dz_initial, dzphimax, mr, nuc, NONLINEARITY)

        if PLASMA
            plasma_equation_local = plasma_equation
        else
            plasma_equation_local = Dict()
        end
        model_keys = (
            NONLINEARITY=NONLINEARITY,
            PLASMA=PLASMA,
            KPARAXIAL=true,
            QPARAXIAL=true,
            ALG=ALG,
        )
        p_model = (responses_local, plasma_equation_local, model_keys)
    elseif geometry == "RT"
        z_local = FloatGPU(z / zu)
        zmax_local = FloatGPU(zmax)
        lam0_local = FloatGPU(lam0)
        dz_plothdf_local = FloatGPU(dz_plothdf)

        p_unit = (ru, zu, tu, Iu, rhou)
        p_grid = (FloatGPU(rmax), Nr, FloatGPU(tmin), FloatGPU(tmax), Nt)
        p_field = (FloatGPU(lam0), initial_condition, HTLOAD, file_ht)
        p_guard = (FloatGPU(rguard), FloatGPU(tguard), FloatGPU(kguard), FloatGPU(wguard))
        p_dzadaptive = (dz_initial, dzphimax, mr, nuc, NONLINEARITY)

        if PLASMA
            plasma_equation_local = plasma_equation
        else
            plasma_equation_local = Dict()
        end
        model_keys = (
            NONLINEARITY=NONLINEARITY,
            PLASMA=PLASMA,
            KPARAXIAL=KPARAXIAL,
            QPARAXIAL=QPARAXIAL,
            ALG=ALG,
        )
        p_model = (responses_local, plasma_equation_local, model_keys)
    elseif geometry == "XY"
        z_local = FloatGPU(z / zu)
        zmax_local = FloatGPU(zmax)
        lam0_local = FloatGPU(lam0)
        dz_plothdf_local = FloatGPU(dz_plothdf)

        p_unit = (xu, yu, zu, Iu)
        p_grid = (FloatGPU(xmin), FloatGPU(xmax), Nx, FloatGPU(ymin), FloatGPU(ymax), Ny)
        p_field = (FloatGPU(lam0), initial_condition)
        p_guard = (FloatGPU(xguard), FloatGPU(yguard), FloatGPU(kxguard), FloatGPU(kyguard))
        p_dzadaptive = (dz_initial, dzphimax, NONLINEARITY)

        model_keys = (
            NONLINEARITY=NONLINEARITY,
            PLASMA=false,
            KPARAXIAL=KPARAXIAL,
            QPARAXIAL=QPARAXIAL,
            ALG=ALG,
        )
        p_model = (responses_local, Dict(), model_keys)
    elseif geometry == "XYT"
        z_local = FloatGPU(z / zu)
        zmax_local = FloatGPU(zmax)
        lam0_local = FloatGPU(lam0)
        dz_plothdf_local = FloatGPU(dz_plothdf)

        p_unit = (xu, yu, zu, tu, Iu, rhou)
        p_grid = (FloatGPU(xmin), FloatGPU(xmax), Nx,
                  FloatGPU(ymin), FloatGPU(ymax), Ny,
                  FloatGPU(tmin), FloatGPU(tmax), Nt)
        p_field = (FloatGPU(lam0), initial_condition)
        p_guard = (FloatGPU(xguard), FloatGPU(yguard), FloatGPU(tguard),
                   FloatGPU(kxguard), FloatGPU(kyguard), FloatGPU(wguard))
        p_dzadaptive = (dz_initial, dzphimax, mr, nuc, NONLINEARITY)

        if PLASMA
            plasma_equation_local = plasma_equation
        else
            plasma_equation_local = Dict()
        end
        model_keys = (
            NONLINEARITY=NONLINEARITY,
            PLASMA=PLASMA,
            KPARAXIAL=KPARAXIAL,
            QPARAXIAL=QPARAXIAL,
            ALG=ALG,
        )
        p_model = (responses_local, plasma_equation_local, model_keys)
    else
        error("Wrong grid geometry.")
    end

    input = Dict(
        "prefix" => prefix,
        "geometry" => geometry,
        "z" => z_local,
        "zmax" => zmax_local,
        "lam0" => lam0_local,
        "Istop" => Istop,
        "dz_plothdf" => dz_plothdf_local,
        "p_unit" => p_unit,
        "p_grid" => p_grid,
        "p_field" => p_field,
        "p_medium" => p_medium,
        "p_guard" => p_guard,
        "p_model" => p_model,
        "p_dzadaptive" => p_dzadaptive,
        "p_info" => p_info,
    )
end


end
