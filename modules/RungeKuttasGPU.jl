module RungeKuttasGPU

    import CuArrays

    const FloatGPU = Float32
    const ComplexGPU = ComplexF32


    struct RungeKutta2
        k1 :: Array{ComplexF64, 2}
        k2 :: Array{ComplexF64, 2}
    end


    struct RungeKutta3
        k1_gpu :: CuArrays.CuArray{ComplexGPU, 2}
        k2_gpu :: CuArrays.CuArray{ComplexGPU, 2}
        k3_gpu :: CuArrays.CuArray{ComplexGPU, 2}
        res_gpu :: CuArrays.CuArray{ComplexGPU, 2}
        dum_gpu :: CuArrays.CuArray{ComplexGPU, 2}
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
            k1_gpu = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k2_gpu = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            k3_gpu = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            res_gpu = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            dum_gpu = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))
            RK = RungeKutta3(k1_gpu, k2_gpu, k3_gpu, res_gpu, dum_gpu)
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
                              f_gpu::CuArrays.CuArray{ComplexGPU, 2},
                              h::FloatGPU, func!::Function)
        func!(f_gpu, RK.res_gpu)
        @inbounds @. RK.k1_gpu = h * RK.res_gpu

        @inbounds @. RK.dum_gpu = f_gpu + 0.5 * RK.k1_gpu
        func!(RK.dum_gpu, RK.res_gpu)
        @inbounds @. RK.k2_gpu = h * RK.res_gpu

        @inbounds @. RK.dum_gpu = f_gpu - RK.k1_gpu + 2. * RK.k2_gpu
        func!(RK.dum_gpu, RK.res_gpu)
        @inbounds @. RK.k3_gpu = h * RK.res_gpu

        @inbounds @. f_gpu = f_gpu + (RK.k1_gpu + 4. * RK.k2_gpu + RK.k3_gpu) / 6.
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
