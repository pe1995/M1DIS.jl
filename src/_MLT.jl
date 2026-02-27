# ============================================================================
# MLT computation of convective quantities
# ============================================================================

"""
    update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5, Teff=5777.0)

Compute MLT parameters F_conv and g_turb based on Gustafsson et al. (1970).
"""
function calc_mlt_local(T_local, P_local, ∇_local, eos_extended, g_surf, alpha_mlt)
    lnpgas_local = log(P_local - (4.0 * σ_SB / (3.0 * c_light)) * (T_local ^ 4))
    lnt_local = log(T_local)
    #lnrho_local = TSO.extended_lookup(eos_extended, :lnRho, lnpgas_local, lnt_local)  
    #lnrho_local, = sample(eos_extended, (:lnRho,), lnpgas_local, lnt_local)  
    
    #=κ_ross = exp(TSO.extended_lookup(eos_extended, :lnRoss, lnrho_local, lnt_local))
    Cp = TSO.extended_lookup(eos_extended, :cₚ, lnrho_local, lnt_local)
    Q = TSO.extended_lookup(eos_extended, :Q, lnrho_local, lnt_local)
    ∇ₐ = TSO.extended_lookup(eos_extended, :∇ₐ, lnrho_local, lnt_local)
    χr = TSO.extended_lookup(eos_extended, :χᵨ, lnrho_local, lnt_local)
    χt = TSO.extended_lookup(eos_extended, :χₜ, lnrho_local, lnt_local)=#

    lnrho_local, lnκ_ross, Cp, Q, ∇ₐ, χr, χt = sample(eos_extended, (:lnRho,:lnRoss, :cₚ, :Q, :∇ₐ, :χᵨ, :χₜ), lnpgas_local, lnt_local)
    κ_ross = exp(lnκ_ross)
    Hp = P_local / (exp(lnrho_local) * g_surf)

    super_adi = ∇_local - ∇ₐ
    if super_adi < 1e-6
        return 0.0, 0.0
    end
    
    # Optically thick limit approximation for Gamma1
    Γ₁_approx = χr / (1 - χt * ∇ₐ)
    c_sound = sqrt(Γ₁_approx * P_local / exp(lnrho_local))
    v_scale = sqrt(g_surf * Q * Hp / 8.0)
    
    # U = (24 sqrt(2) sigma T^3) / (kappa rho Hp alpha rho Cp v_scale)
    numerator = 24.0 * sqrt(2.0) * σ_SB * T_local^3
    denominator = κ_ross * exp(lnrho_local) * Hp * alpha_mlt * exp(lnrho_local) * Cp * v_scale
    U = numerator / denominator

    # Solve cubic for efficiency factor xi
    # 2Uξ³ + ξ² + Uξ - (∇ - ∇ad) = 0
    xi = 0.5
    for _ in 1:50
        xi_sq = xi^2
        f_val = 2.0 * U * xi_sq * xi + xi_sq + U * xi - super_adi
        df_dz = 6.0 * U * xi_sq + 2.0 * xi + U
        dxi = f_val / df_dz
        xi -= dxi
        if abs(dxi) < 1e-6 * xi; break; end
    end
    xi = max(xi, 1e-9)

    v_real = v_scale * xi
    ratio = v_real / c_sound
    soft_factor = (1.0 + ratio^4)^0.25
    v_real = v_real / soft_factor
    xi = xi / soft_factor

    Flux = (0.5 * alpha_mlt) * (exp(lnrho_local) * Cp * T_local) * v_scale * xi^3
    return Flux, v_real
end

function update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; 
    alpha_mlt=1.5, Teff=5777.0, v_mic=0.0)
    n_depth = length(T)
    # Reset outputs
    fill!(F_conv, 0.0)
    fill!(v_conv, 0.0)
    fill!(dFconv_dT, 0.0)
    
    @inbounds for n in 2:n_depth
        P_rad_n = (4.0 * σ_SB / (3.0 * c_light)) * (T[n] ^ 4)
        P_tot_n = P_gas[n] + P_rad_n
        
        P_rad_nm1 = (4.0 * σ_SB / (3.0 * c_light)) * (T[n-1] ^ 4)
        P_tot_nm1 = P_gas[n-1] + P_rad_nm1

        # Calculate Gradient (Backward Difference)
        dlnT = log(T[n] / T[n-1])
        dlnP = log(P_tot_n / P_tot_nm1)
        ∇_base = dlnT / dlnP
        
        # Base Flux
        F_base, v_base = calc_mlt_local(T[n], P_tot_n, ∇_base, eos_extended, g_surf, alpha_mlt)
        F_conv[n] = F_base
        v_conv[n] = v_base

        # Calculate Derivative dF/dT
        delta_T = 0.001 * T[n]
        T_pert = T[n] + delta_T
        dlnT_pert = log(T_pert / T[n-1])
        ∇_pert = dlnT_pert / dlnP
        F_pert, _ = calc_mlt_local(T_pert, P_tot_n, ∇_pert, eos_extended, g_surf, alpha_mlt)
        
        # Stability fix (Gustafsson et al.)
        if F_base <= 1e-10
            b = 0.005 
            T_recipe = T[n] * (1.0 + b)
            
            dlnT_recipe = log(T_recipe / T[n-1])
            ∇_recipe = dlnT_recipe / dlnP
            F_recipe, _ = calc_mlt_local(T_recipe, P_tot_n, ∇_recipe, eos_extended, g_surf, alpha_mlt)
            
            if F_recipe > 1e-10
                dFconv_dT[n] = (F_recipe) / (T_recipe - T[n])
            else
                dFconv_dT[n] = 0.0
            end
        else
            dFconv_dT[n] = (F_pert - F_base) / delta_T
        end
    end
    
    F_conv[1] = F_conv[2]
    dFconv_dT[1] = dFconv_dT[2]

    # Turbulent Pressure and Gravity
    prev_P_turb = 0.5 * ρ[1] * (v_conv[1]^2 + v_mic^2)
    #prev_kappa = exp(TSO.extended_lookup(eos_extended, :lnRoss, log(ρ[1]), log(T[1])))
    prev_kappa, = sample(eos_extended, (:lnRoss,), log(ρ[1]), log(T[1])) .|> exp

    curr_P_turb = 0.5 * ρ[2] * (v_conv[2]^2 + v_mic^2)
    #curr_kappa = exp(TSO.extended_lookup(eos_extended, :lnRoss, log(ρ[2]), log(T[2])))
    curr_kappa, = sample(eos_extended, (:lnRoss,), log(ρ[2]), log(T[2])) .|> exp
    
    dP_dtau = (curr_P_turb - prev_P_turb) / (τ_ross[2] - τ_ross[1])
    g_turb[1] = prev_kappa * dP_dtau
    
    @inbounds for i in 2:n_depth-1
        next_P_turb = 0.5 * ρ[i+1] * (v_conv[i+1]^2 + v_mic^2)
        
        h1 = τ_ross[i] - τ_ross[i-1]
        h2 = τ_ross[i+1] - τ_ross[i]
        dP_dtau = - (h2 / (h1 * (h1 + h2))) * prev_P_turb +
                  ((h2 - h1) / (h1 * h2)) * curr_P_turb +
                  (h1 / (h2 * (h1 + h2))) * next_P_turb
                  
        g_turb[i] = curr_kappa * dP_dtau
        
        prev_P_turb = curr_P_turb
        curr_P_turb = next_P_turb
        #curr_kappa = exp(TSO.extended_lookup(eos_extended, :lnRoss, log(ρ[i+1]), log(T[i+1])))
        curr_kappa, = sample(eos_extended, (:lnRoss,), log(ρ[i+1]), log(T[i+1])) .|> exp
    end
    
    dP_dtau = (curr_P_turb - prev_P_turb) / (τ_ross[n_depth] - τ_ross[n_depth-1])
    g_turb[n_depth] = curr_kappa * dP_dtau

    gturb_stabilizer!(g_turb, g_surf)
end

"""
    gturb_stabilizer!(g_turb, g_surf; max_fraction=0.1, passes=5, relax=0.2)

Stabilisiert g_turb in-place. Nutzt den aktuellen Inhalt von g_turb als 
Zielwert und relaxiert ihn gegen den vorherigen Zustand.
"""
function gturb_stabilizer!(g_turb, g_surf; max_fraction=0.1, passes=5, relax=0.2)
    n = length(g_turb)
    if n < 3 return end

    g_old = copy(g_turb) 
    
    g_tmp = zeros(n)
    for _ in 1:passes
        g_tmp[1] = 0.75 * g_turb[1] + 0.25 * g_turb[2]
        g_tmp[n] = 0.75 * g_turb[n] + 0.25 * g_turb[n-1]
        @inbounds for i in 2:n-1
            g_tmp[i] = 0.25 * g_turb[i-1] + 0.5 * g_turb[i] + 0.25 * g_turb[i+1]
        end
        g_turb .= g_tmp
    end

    m = g_turb .* g_old .< 0.0
    g_turb[m] .= g_turb[m] .* 0.5
    #=@inbounds for i in 1:n
        g_turb[i] = (1.0 - relax) * g_old[i] + relax * g_turb[i]
    end=#

    limit = abs(g_surf * max_fraction)
    @inbounds for i in 1:n
        g_turb[i] = limit * tanh(g_turb[i] / limit)
    end
end

# ============================================================================
# Temperature structure adjustment (non-Feutrier, old and unreliable)
# ============================================================================

"""
    update_temperature_correction_robust!(dT, F_rad, F_conv, dFconv_dT, T, τ_grid, Teff, J; damping=0.5)

A robust temperature correction scheme based on the Unsold-Mawe (Flux-scaling) procedure,
augmented with a small ALI term strictly for the surface layers.
"""
function update_temperature_correction_robust!(dT, F_rad, F_conv, dFconv_dT, T, τ_grid, Teff, J; damping=0.5)
    F_target = σ_SB * Teff^4
    F_tot = max.(F_rad .+ F_conv, 1e-12)
    n_depth = length(T)

    # 1. Flux Scaling Factor
    #gain = 0.5 
    
    #safe_F_target = max(F_target, 1e-10)
    #ratio_raw = (F_tot ./ safe_F_target)
    
    #ratio = 1.0 .+ gain .* (ratio_raw .- 1.0)
    #ratio .= clamp.(ratio, 0.5, 2.0)
    ratio = F_tot ./ F_target
    
    # 3. Integrate new optical depth scale
    τ_new = similar(τ_grid)
    τ_new[1] = τ_grid[1] * ratio[1]
    @inbounds for k in 2:n_depth
        dτ = τ_grid[k] - τ_grid[k-1]
        r_avg = 0.5 * (ratio[k] + ratio[k-1])
        τ_new[k] = τ_new[k-1] + r_avg * dτ
    end

    # 4. Interpolate T to new grid (Unsold-Mawe)
    sp = sortperm(τ_new)
    interp = linear_interpolation(Interpolations.deduplicate_knots!(log.(τ_new[sp])), log.(T[sp]), extrapolation_bc=Line())
    log_T_new = interp(log.(τ_grid))
    T_new = exp.(log_T_new)
    dT_new = T_new .- T

    dT_surf = similar(dT_new)
    surf_err = (F_target - F_rad[1])/F_target * 0.25 * T[1]
    fill!(dT_surf, surf_err) 
    
    dT_new .+= dT_surf
    m = (dT_new .* dT) .< 0
    dT_new[m] .*= 0.75
    dT .= clamp.(dT_new, -0.05.*T, 0.05.*T)

    #=#dT_mawe = T_new .- T

    local_ratio = abs.(F_tot ./ F_target)
    corr_factor_local = (1.0 ./ local_ratio) .^ 0.25
    blend = exp.(-1.0*τ_grid)
    f = (1.0 .+ blend .* (corr_factor_local .- 1.0))
    T_new .= T_new .* f
    dT_mawe = (T_new .- T)=#
    
    # Combined Correction
    #damp_depth = (1.0 .+ (damping .- 1.0) .* exp.(-1.0 ./ τ_grid))
    #dT .= damp_depth .* dT_new
end



