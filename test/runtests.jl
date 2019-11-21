import DelimitedFiles
using Test

PROJECT = joinpath(@__DIR__, "..")

@testset "diffraction" begin
    include("test_diffraction_rt.jl")
end

# @testset "Run examples" begin
#     examples = ["R", "T", "XY", "RT", "RT_lattice"]
#     for example in examples
#         cd(joinpath(PROJECT, "examples", example))
#         proc = run(
#             `julia -O3 --check-bounds=no $PROJECT/src/Filament.jl input.jl`
#         )
#         @test proc.exitcode == 0
#     end
# end
