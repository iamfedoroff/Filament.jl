module Guards

import Units
import Grids
import Media


struct GuardFilter
    R :: Array{Float64, 1}
    T :: Array{Float64, 1}
    K :: Array{Float64, 2}
    W :: Array{Float64, 1}
end


function GuardFilter(unit::Units.Unit, grid::Grids.Grid, medium::Media.Medium,
                     rguard_width::Float64, tguard_width::Float64,
                     kguard::Float64, wguard::Float64)
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard_width, mode="right")

    # Temporal guard filter:
    Tguard = guard_window(grid.t, tguard_width, mode="both")

    # Angular guard filter:
    k = Media.k_func.(medium, grid.w * unit.w)
    kmax = k * sind(kguard)
    Kguard = zeros((grid.Nr, grid.Nw))
    for j=2:grid.Nw   # from 2 because kmax[1]=0 since w[1]=0
        for i=1:grid.Nr
            if kmax[j] != 0.
                Kguard[i, j] = exp(-((grid.k[i] * unit.k)^2 / kmax[j]^2)^20)
            end
        end
    end

    # Spectral guard filter:
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)

    return GuardFilter(Rguard, Tguard, Kguard, Wguard)
end



"""
Lossy guard window at the ends of grid coordinate.

    x: grid coordinate
    guard_width: the width of the lossy guard window
    mode: "left" - lossy guard only on the left end of the grid
          "right" - lossy guard only on the right end of the grid
          "both" - lossy guard on both ends of the grid
"""
function guard_window(x::Array{Float64, 1}, guard_width::Float64; mode="both")
    assert(guard_width >= 0.)
    assert(mode in ["left", "right", "both"])

    if mode in ["left", "right"]
        assert(guard_width <= x[end] - x[1])
    else
        assert(guard_width <= 0.5 * (x[end] - x[1]))
    end

    Nx = length(x)

    if guard_width == 0
        guard = ones(Nx)
    else
        width = 0.5 * guard_width

        # Left guard
        guard_xmin = x[1]
        guard_xmax = x[2] + guard_width
        gauss1 = zeros(Nx)
        gauss2 = ones(Nx)
        for i=1:Nx
            if x[i] >= guard_xmin
                gauss1[i] = 1. - exp(-((x[i] - guard_xmin) / width)^6)
            end
            if x[i] <= guard_xmax
                gauss2[i] = exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_left = 0.5 * (gauss1 + gauss2)

        # Right guard
        guard_xmin = x[end] - guard_width
        guard_xmax = x[end]
        gauss1 = ones(Nx)
        gauss2 = zeros(Nx)
        for i=1:Nx
            if x[i] >= guard_xmin
                gauss1[i] = exp(-((x[i] - guard_xmin) / width)^6)
            end
            if x[i] <= guard_xmax
                gauss2[i] = 1. - exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_right = 0.5 * (gauss1 + gauss2)

        # Result guard:
        if mode == "left"
            guard = guard_left
        elseif mode == "right"
            guard = guard_right
        elseif mode == "both"
            guard = guard_left + guard_right - 1.
        end
    end
    return guard
end


end
