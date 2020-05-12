import DelimitedFiles
using Test

import Filament

@testset "physics" begin
    @testset "diffraction" begin
        @testset "r" begin
            include("test_diffraction_r.jl")
        end
        @testset "rt" begin
            include("test_diffraction_rt.jl")
        end
        @testset "xy" begin
            include("test_diffraction_xy.jl")
        end
    end

    @testset "dispersion" begin
        @testset "t" begin
            include("test_dispersion_t.jl")
        end
        @testset "rt" begin
            include("test_dispersion_rt.jl")
        end
    end
end
