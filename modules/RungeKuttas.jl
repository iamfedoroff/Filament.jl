module RungeKuttas

    import CuArrays

    const FloatGPU = Float32
    const ComplexGPU = ComplexF32


    struct RungeKutta2
        k1 :: Array{ComplexF64, 2}
        k2 :: Array{ComplexF64, 2}
    end


    struct RungeKutta3
        k1 :: CuArrays.CuArray{ComplexGPU, 2}
        k2 :: CuArrays.CuArray{ComplexGPU, 2}
        k3 :: CuArrays.CuArray{ComplexGPU, 2}
        tmp :: CuArrays.CuArray{ComplexGPU, 2}
    end


    struct RungeKutta4
        k1 :: Array{ComplexF64, 2}
        k2 :: Array{ComplexF64, 2}
        k3 :: Array{ComplexF64, 2}
        k4 :: Array{ComplexF64, 2}
    end


    function RungeKutta(order, Nr, Nw)
        if order == 2
            k1 = zeros(ComplexF64, (Nr, Nw))
            k2 = zeros(ComplexF64, (Nr, Nw))
            RK = RungeKutta2(k1, k2)
        elseif order == 3
            k1 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k3 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            tmp = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            RK = RungeKutta3(k1, k2, k3, tmp)
        elseif order == 4
            k1 = zeros(ComplexF64, (Nr, Nw))
            k2 = zeros(ComplexF64, (Nr, Nw))
            k3 = zeros(ComplexF64, (Nr, Nw))
            k4 = zeros(ComplexF64, (Nr, Nw))
            RK = RungeKutta4(k1, k2, k3, k4)
        else
            print("Wrong Runge-Kutta order\n")
            quit()
        end
        return RK
    end


    function RungeKutta_calc!(RK::RungeKutta2, f::Array{ComplexF64, 2},
                              h::Float64, func::Function)
        dum = func(f)
        @inbounds @. RK.k1 = h * dum

        @inbounds @. dum = f + 2. / 3. * RK.k1
        dum = func(dum)
        @inbounds @. RK.k2 = h * dum

        @inbounds @. f = f + (RK.k1 + 3. * RK.k2) / 4.
        return nothing
    end


    function RungeKutta_calc!(RK::RungeKutta3,
                              f::CuArrays.CuArray{ComplexGPU, 2},
                              h::FloatGPU,
                              func!::Function)
        func!(f, RK.k1)
        @inbounds @. RK.k1 = h * RK.k1

        @inbounds @. RK.tmp = f + 0.5 * RK.k1
        func!(RK.tmp, RK.k2)
        @inbounds @. RK.k2 = h * RK.k2

        @inbounds @. RK.tmp = f - RK.k1 + 2. * RK.k2
        func!(RK.tmp, RK.k3)
        @inbounds @. RK.k3 = h * RK.k3

        @inbounds @. f = f + (RK.k1 + 4. * RK.k2 + RK.k3) / 6.
        return nothing
    end


    function RungeKutta_calc!(RK::RungeKutta4, f::Array{ComplexF64, 2},
                              h::Float64, func::Function)
        dum = func(f)
        @inbounds @. RK.k1 = h * dum

        @inbounds @. dum = f + 0.5 * RK.k1
        dum = func(dum)
        @inbounds @. RK.k2 = h * dum

        @inbounds @. dum = f + 0.5 * RK.k2
        dum = func(dum)
        @inbounds @. RK.k3 = h * dum

        @inbounds @. dum = f + RK.k3
        dum = func(dum)
        @inbounds @. RK.k4 = h * dum

        @inbounds @. f = f + (RK.k1 + 2. * RK.k2 + 2. * RK.k3 + RK.k4) / 6.
        return nothing
    end


end
