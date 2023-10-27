

function IController(;qmin = 1//5,qmax = 10,qsteady_min = 1,qsteady_max = 1,gamma = 9 //10, qold = 1 // 10^4)
    QT = eltype(qmin)

    return IController{QT}(qmin,QT(qmax),QT(qsteady_min),QT(qsteady_max),QT(gamma),QT(qold))
end

function update_EEst!(cache::PseudoTransientCache)
    @unpack u,uprev,uprev2, alpha,alpha_prev,atmp,tmp = cache

    #=uprev = cache.uprev
    uprev2 = cache.uprev2
    alpha_prev = cache.alpha_prev
    alpha = cache.alpha=#

    alpha1 = alpha*(alpha + alpha_prev) #dt1 #alpha_prev = t - tprev
    alpha2 = alpha_prev*(alpha + alpha_prev) #dt2
    c = 7/12
    r = c*alpha^2
    
    @.. broadcast=false tmp =r * cache.internalnorm((u - uprev) / alpha1 -
                                                             (uprev - uprev2) / alpha2)

    #=if cache.stats.nsteps % 10 == 0
        push!(alpha_contain,alpha)
    end=#
    calculate_residuals!(atmp,tmp,uprev,u,cache.abstol,cache.abstol,(u,t) -> cache.internalnorm(u),1.0)
    cache.EEst = cache.internalnorm(atmp)
end


function accept_step_controller(cache::PseudoTransientCache,controller::IController)
    return cache.EEst <= 1

end

function update_alpha!(cache::PseudoTransientCache,controller::IController)
    q = stepsize_controller!(cache, controller)
    if accept_step_controller(cache,controller)
        cache.alpha *= step_accept_controller!(cache,controller,q)
    end

end

function stepsize_controller!(cache::PseudoTransientCache,controller::IController)
    @unpack qmin, qmax, gamma = controller
    EEst = DiffEqBase.value(cache.EEst)

    if iszero(EEst)
        q = inv(qmax)
    else
        #expo = 1 / (get_current_adaptive_order(alg, integrator.cache) + 1)
        qtmp = EEst / gamma
        @fastmath q = DiffEqBase.value(max(inv(qmax), min(inv(qmin), qtmp)))
        # TODO: Shouldn't this be in `step_accept_controller!` as for the PI controller?
        controller.qold = DiffEqBase.value(cache.alpha) / q
    end
    return q
end

function step_accept_controller!(cache, controller::IController, q)
    @unpack qsteady_min, qsteady_max = controller

    if qsteady_min <= q <= qsteady_max
        q = one(q)
    end
    return cache.alpha / q # new dt
end

function step_reject_controller!(cache, controller::IController)
    @unpack qold = controller
    cache.alpha = qold
end
