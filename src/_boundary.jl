# ============================================================================
# Hydrostatic equilibrium boundary conditions (top)
# ============================================================================

function lnP_boundary(T_top, g_eff_top, eos, τ_top; maxiter=400, tol=1e-9, P_guess=1e-4)
    lnT = log(T_top)
    C = log(g_eff_top * τ_top)

    function calc_f(lp)
        lnρ, lnκ_ross = sample(eos, (:lnRho, :lnRoss), lp, lnT)

        lp_eps = lp + 0.001
        _, lnκ_eps = sample(eos, (:lnRho, :lnRoss), lp_eps, lnT)
        alpha = (lnκ_eps - lnκ_ross) / 0.001
        alpha = clamp(alpha, -0.5, 2.5)
        return lp + lnκ_ross - C - log(1.0 + alpha)
    end

    lp = log(P_guess)
    f = calc_f(lp)
    if abs(f) < tol
        return lp
    end

    lp1, f1 = lp, f
    lp2, f2 = lp, f

    step = 2.0
    for _ in 1:50
        if f1 > 0
            lp1 -= step
            f1 = calc_f(lp1)
        end
        if f2 < 0
            lp2 += step
            f2 = calc_f(lp2)
        end
        if f1 < 0 && f2 > 0
            break
        end
        step *= 1.5
    end

    if f1 > 0 || f2 < 0
        @verbose_warn 1 "Could not bracket the root in lnP_boundary (f1=$f1, f2=$f2)"
        return lp
    end

    for i in 1:maxiter
        lp_mid = 0.5 * (lp1 + lp2)
        f_mid = calc_f(lp_mid)

        if abs(f_mid) < tol || (lp2 - lp1) < tol
            return lp_mid
        end

        if f_mid < 0
            lp1, f1 = lp_mid, f_mid
        else
            lp2, f2 = lp_mid, f_mid
        end
    end

    @verbose_warn 1 "Top pressure did not converge after $(maxiter) bisection iterations; using last iterate"
    return 0.5 * (lp1 + lp2)
end

# ============================================================================
# enforce adiabatic bottom boundary condition
# ============================================================================

function force_adiabatic_bottom!(T, P, eos_extended; n_force=5)
    n_depth = length(T)
    start_idx = n_depth - n_force + 1

    for i in start_idx:n_depth
        lnP_prev = log(P[i-1])
        lnT_prev = log(T[i-1])

        lnRho, ∇_ad = sample(eos_extended, (:lnRho, :∇ₐ), lnP_prev, lnT_prev)

        dlnP = log(P[i]) - lnP_prev
        T[i] = T[i-1] * exp(∇_ad * dlnP)

        if T[i] < T[i-1]
             T[i] = T[i-1] + 1e-4
        end
    end
end

# ============================================================================
# External irradiation boundary conditions
# ============================================================================

function irradiate(eos, opa::TSO.EquationOfState, T_irradiation, R_irradiation, d_irradiation, F_irradiation)
    S = if !isnothing(F_irradiation)
        F_irradiation ./ π
    elseif _has_source(opa)
        rho_min, rho_max = TSO.limits(eos, 2)
        rho_irr = exp((rho_max + rho_min) / 2)
        sample(merge(eos, opa), (:src,), Float64(log(rho_irr)), Float64(log(T_irradiation)))[1]
    else
        TSO.Bλ_fast.(opa.opacity.λ, Float64(T_irradiation))
    end
    S .= S .* (R_irradiation ./ d_irradiation) .^2 .* opa.opacity.weights

    S
end
