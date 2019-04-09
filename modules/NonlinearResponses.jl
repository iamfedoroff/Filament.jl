module NonlinearResponses


    struct NonlinearResponse{T}
        Rnl :: T
        func :: Function
        p :: Tuple
    end


    function calculate!(nresp::NonlinearResponse, z, F, E)
        nresp.func(z, F, E, nresp.p)
        return nothing
    end


end
