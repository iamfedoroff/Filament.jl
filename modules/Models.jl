module Models

using PyCall
@pyimport scipy.constants as sc
@pyimport matplotlib.pyplot as plt

import Media

C0 = sc.c   # speed of light in vacuum
EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
MU0 = sc.mu_0   # the magnetic constant [N/A^2]
QE = sc.e   # elementary charge [C]
ME = sc.m_e   # electron mass [kg]
HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


struct Model
    KZ :: Array{Float64, 2}
end


function Model(unit, grid, field, medium, keys)

    rguard_width = 1.
    tguard_width = 1.

    Rguard = guard_window(grid.r, rguard_width, mode="right")
    Tguard = guard_window(grid.t, tguard_width, mode='both')


    plt.figure(dpi=300)
    plt.plot(grid.r, Rguard)
    plt.tight_layout()
    plt.show()
    quit()

    # KZ = zeros((grid.Nx, grid.Ny))
    # for j = 1:grid.Ny
    #     for i = 1:grid.Nx
    #         KZ[i, j] = -((grid.kx[i] * unit.kx)^2 +
    #                      (grid.ky[j] * unit.ky)^2) / (2. * k0) * unit.z
    #     end
    # end
    #
    # return Model(KZ)
end


# function zstep(z, field, model)
#     S = fft(field.E)
#     S = @. S * exp(1im * model.KZ * z)
#     field.E = ifft(S)
# end


"""
Lossy guard window at the ends of grid coordinate.

    x: grid coordinate
    guard_width: the width of the lossy guard window
    mode: "left" - lossy guard only on the left end of the grid
          "right" - lossy guard only on the right end of the grid
          "both" - lossy guard on both ends of the grid
"""
function guard_window(x, guard_width; mode="both")
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
