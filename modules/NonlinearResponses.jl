module NonlinearResponses

import Units
import Grids
import Fields
import Media


struct NonlinearResponse{T}
    Rnl :: T
    calc :: Function
    p_calc :: Tuple
    dzadapt :: Function
    p_dzadapt :: Tuple
end


function init(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
              medium::Media.Medium, responses_list)
    responses = []
    for item in responses_list
        init = item["init"]
        Rnl, calc, p_calc, dzadapt, p_dzadapt = init(unit, grid, field, medium, item)
        response = NonlinearResponse(Rnl, calc, p_calc, dzadapt, p_dzadapt)
        push!(responses, response)
    end
    return tuple(responses...)
end


function calculate!(nresp::NonlinearResponse, z::AbstractFloat,
                    F::AbstractArray, E::AbstractArray)
    nresp.calc(z, F, E, nresp.p_calc)
    return nothing
end


function dzadaptive(nresp::NonlinearResponse, phimax::AbstractFloat)
    return nresp.dzadapt(phimax, nresp.p_dzadapt)
end


end
