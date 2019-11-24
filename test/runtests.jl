import DelimitedFiles
using Test

PROJECT = joinpath(@__DIR__, "..")

@testset "diffraction" begin
    include("test_diffraction_r.jl")
    include("test_diffraction_rt.jl")
    include("test_diffraction_xy.jl")
end
