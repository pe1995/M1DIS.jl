#= boundary conditions =#

"""
    lnP_boundary(T_top, g_eff_top, eos, τ_top; maxiter=200, tol=1e-8, P_guess=1e-4)

Compute the upper boundary condition pressure iteratively from the temperature and optical depth,
assuming that P(z=0) = 0 and the pressure varies linearly until τ_top.
"""
function lnP_boundary(T_top, g_eff_top, eos, τ_top; maxiter=200, tol=1e-8, P_guess=1e-4)
    lnP = log(P_guess) 
	lnT = log(T_top)
    for _ in 1:maxiter
        lnρ = TSO.extended_lookup(eos, :lnRho, lnP, lnT)
        κ = exp(TSO.extended_lookup(eos, :lnRoss, lnρ, lnT))

        P_new = g_eff_top * τ_top / κ
        lnP_new = log(P_new)

        if abs(lnP_new - lnP) < tol
            return lnP_new
        end

        lnP = 0.5*(lnP + lnP_new)   # damping ensures stability
    end

    @warn "Top pressure did not converge after $(maxiter) iterations; using last iterate"
    return lnP
end

function force_adiabatic_bottom!(T, P, eos_extended; n_force=5)
    n_depth = length(T)
    start_idx = n_depth - n_force + 1
    
    for i in start_idx:n_depth
        lnP_prev = log(P[i-1])
        lnT_prev = log(T[i-1])
        
        lnRho = TSO.extended_lookup(eos_extended, :lnRho, lnP_prev, lnT_prev)
        ∇_ad  = TSO.extended_lookup(eos_extended, :∇ₐ, lnRho, lnT_prev)
        
        dlnP = log(P[i]) - lnP_prev
        T[i] = T[i-1] * exp(∇_ad * dlnP)
        
        if T[i] < T[i-1]
             T[i] = T[i-1] + 1e-4 # Tiny increment fallback
        end
    end
end