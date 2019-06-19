module PFunctions


struct PFunction{F<:Function, T<:Tuple} <: Function
    func :: F
    p :: T
end


function (pfunc::PFunction)(x...)
    pfunc.func(x..., pfunc.p)
end


end
