module Models

import Media


struct Model
    KZ :: Array{Float64, 2}
end


function Model(unit, grid, field, medium)
    k0 = Media.wave_number(medium, field.w0)

    KZ = zeros((grid.Nx, grid.Ny))
    for j = 1:grid.Ny
        for i = 1:grid.Nx
            KZ[i, j] = -((grid.kx[i] * unit.kx)^2 +
                         (grid.ky[j] * unit.ky)^2) / (2. * k0) * unit.z
        end
    end

    return Model(KZ)
end


function zstep(z, field, model)
    S = fft(field.E)
    S = @. S * exp(1im * model.KZ * z)
    field.E = ifft(S)
end


end
