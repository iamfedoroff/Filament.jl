push!(LOAD_PATH, joinpath("..", "..", "src"))
import Filament

fname = "input.jl"

input = Filament.prepare(fname)

Filament.run(input)
