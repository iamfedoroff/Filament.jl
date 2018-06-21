module RungeKuttas


    mutable struct RungeKutta2
        k1 :: Array{Complex128, 2}
        k2 :: Array{Complex128, 2}
    end


    mutable struct RungeKutta3
        k1 :: Array{Complex128, 2}
        k2 :: Array{Complex128, 2}
        k3 :: Array{Complex128, 2}
    end


    mutable struct RungeKutta4
        k1 :: Array{Complex128, 2}
        k2 :: Array{Complex128, 2}
        k3 :: Array{Complex128, 2}
        k4 :: Array{Complex128, 2}
    end


    function RungeKutta(order, Nr, Nw)
        if order == 2
            k1 = zeros(Complex128, (Nr, Nw))
            k2 = zeros(Complex128, (Nr, Nw))
            RK = RungeKutta2(k1, k2)
        elseif order == 3
            k1 = zeros(Complex128, (Nr, Nw))
            k2 = zeros(Complex128, (Nr, Nw))
            k3 = zeros(Complex128, (Nr, Nw))
            RK = RungeKutta3(k1, k2, k3)
        elseif order == 4
            k1 = zeros(Complex128, (Nr, Nw))
            k2 = zeros(Complex128, (Nr, Nw))
            k3 = zeros(Complex128, (Nr, Nw))
            k4 = zeros(Complex128, (Nr, Nw))
            RK = RungeKutta4(k1, k2, k3, k4)
        else
            print("Wrong Runge-Kutta order\n")
            quit()
        end
        return RK
    end


    function RungeKutta_calc!(RK::RungeKutta2, f, h, func)
        dum = func(f)
        @inbounds @. RK.k1 = h * dum

        @inbounds @. dum = f + 2. / 3. * RK.k1
        dum = func(dum)
        @inbounds @. RK.k2 = h * dum

        @inbounds @. f = f + (RK.k1 + 3. * RK.k2) / 4.
        return nothing
    end


    function RungeKutta_calc!(RK::RungeKutta3, f, h, func)
        dum = func(f)
        @inbounds @. RK.k1 = h * dum

        @inbounds @. dum = f + 0.5 * RK.k1
        dum = func(dum)
        @inbounds @. RK.k2 = h * dum

        @inbounds @. dum = f - RK.k1 + 2. * RK.k2
        dum = func(dum)
        @inbounds @. RK.k3 = h * dum

        @inbounds @. f = f + (RK.k1 + 4. * RK.k2 + RK.k3) / 6.
        return nothing
    end


    function RungeKutta_calc!(RK::RungeKutta4, f, h, func)
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
