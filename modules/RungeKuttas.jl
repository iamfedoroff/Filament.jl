module RungeKuttas

    import CuArrays

    const FloatGPU = Float32
    const ComplexGPU = ComplexF32


    struct RungeKutta2
        k1 :: CuArrays.CuArray{ComplexGPU, 2}
        k2 :: CuArrays.CuArray{ComplexGPU, 2}
        tmp :: CuArrays.CuArray{ComplexGPU, 2}
    end


    struct RungeKutta3
        k1 :: CuArrays.CuArray{ComplexGPU, 2}
        k2 :: CuArrays.CuArray{ComplexGPU, 2}
        k3 :: CuArrays.CuArray{ComplexGPU, 2}
        tmp :: CuArrays.CuArray{ComplexGPU, 2}
    end


    struct RungeKutta4
        k1 :: CuArrays.CuArray{ComplexGPU, 2}
        k2 :: CuArrays.CuArray{ComplexGPU, 2}
        k3 :: CuArrays.CuArray{ComplexGPU, 2}
        k4 :: CuArrays.CuArray{ComplexGPU, 2}
        tmp :: CuArrays.CuArray{ComplexGPU, 2}
    end


    function RungeKutta(order::Int64, Nr::Int64, Nw::Int64)
        if order == 2
            k1 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            tmp = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            RK = RungeKutta2(k1, k2, tmp)
        elseif order == 3
            k1 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k3 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            tmp = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            RK = RungeKutta3(k1, k2, k3, tmp)
        elseif order == 4
            k1 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k3 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k4 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            tmp = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            RK = RungeKutta4(k1, k2, k3, k4, tmp)
        else
            println("ERROR: Wrong Runge-Kutta order.")
            exit()
        end
        return RK
    end


    function solve!(RK::RungeKutta2, u::CuArrays.CuArray{ComplexGPU, 2},
                    h::FloatGPU, func!::Function, p::Tuple)
        func!(RK.k1, u, p)
        @inbounds @. RK.k1 = h * RK.k1

        @inbounds @. RK.tmp = u + 2. / 3. * RK.k1
        func!(RK.k2, RK.tmp, p)
        @inbounds @. RK.k2 = h * RK.k2

        @inbounds @. u = u + (RK.k1 + 3. * RK.k2) / 4.
        return nothing
    end


    function solve!(RK::RungeKutta3, u::CuArrays.CuArray{ComplexGPU, 2},
                    h::FloatGPU, func!::Function, p::Tuple)
        func!(RK.k1, u, p)
        @inbounds @. RK.k1 = h * RK.k1

        @inbounds @. RK.tmp = u + 0.5 * RK.k1
        func!(RK.k2, RK.tmp, p)
        @inbounds @. RK.k2 = h * RK.k2

        @inbounds @. RK.tmp = u - RK.k1 + 2. * RK.k2
        func!(RK.k3, RK.tmp, p)
        @inbounds @. RK.k3 = h * RK.k3

        @inbounds @. u = u + (RK.k1 + 4. * RK.k2 + RK.k3) / 6.
        return nothing
    end


    function solve!(RK::RungeKutta4, u::CuArrays.CuArray{ComplexGPU, 2},
                    h::FloatGPU, func!::Function, p::Tuple)
        func!(RK.k1, u, p)
        @inbounds @. RK.k1 = h * RK.k1

        @inbounds @. RK.tmp = u + 0.5 * RK.k1
        func!(RK.k2, RK.tmp, p)
        @inbounds @. RK.k2 = h * RK.k2

        @inbounds @. RK.tmp = u + 0.5 * RK.k2
        func!(RK.k3, RK.tmp, p)
        @inbounds @. RK.k3 = h * RK.k3

        @inbounds @. RK.tmp = u + RK.k3
        func!(RK.k4, RK.tmp, p)
        @inbounds @. RK.k4 = h * RK.k4

        @inbounds @. u = u + (RK.k1 + 2. * RK.k2 + 2. * RK.k3 + RK.k4) / 6.
        return nothing
    end


end
