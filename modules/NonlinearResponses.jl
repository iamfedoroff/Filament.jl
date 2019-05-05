module NonlinearResponses

import Units
import Grids
import Fields
import Media


struct NonlinearResponse{T}
    Rnl :: T
    func :: Function
    p :: Tuple
end


function init(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
              medium::Media.Medium, responses_list)
    responses = []
    for item in responses_list
        init = item["init"]
        Rnl, calc, p = init(unit, grid, field, medium, item)
        response = NonlinearResponses.NonlinearResponse(Rnl, calc, p)
        push!(responses, response)
    end
    return responses
end


function calculate!(nresp::NonlinearResponse, z::AbstractFloat,
                    F::AbstractArray, E::AbstractArray)
    nresp.func(z, F, E, nresp.p)
    return nothing
end


end
