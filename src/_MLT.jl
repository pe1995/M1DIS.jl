"""
    update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5, Teff=5777.0)

Compute MLT parameters F_conv and g_turb based on Gustafsson et al. (1970).
"""
function update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5, Teff=5777.0)
    n_depth = length(T)
    # Reset outputs
    F_conv .= 0.0
    v_conv .= 0.0
    g_turb .= 0.0
    dFconv_dT .= 0.0
    
    # Pre-calculate thermodynamic variables (Assume constant during local linearization)
    lnrho = log.(ρ)
    lnT = log.(T)

    P_rad = (4.0 * σ_SB / (3.0 * c_light)) .* (T .^ 4)
    P_tot = P_gas .+ P_rad
    
    function calc_mlt_local(n, T_local, P_local, ∇_local)
        lnpgas_local = log(P_local - (4.0 * σ_SB / (3.0 * c_light)) * (T_local ^ 4))
        lnt_local = log(T_local)
        lnrho_local = TSO.extended_lookup(eos_extended, :lnRho, lnpgas_local, lnt_local)  
        Hp = P_local / (exp(lnrho_local) * g_surf)
        
        κ_ross = exp.(TSO.extended_lookup(eos_extended, :lnRoss, lnrho_local, lnt_local))
        Cp = TSO.extended_lookup(eos_extended, :cₚ, lnrho_local, lnt_local)
        Q = TSO.extended_lookup(eos_extended, :Q, lnrho_local, lnt_local)
        ∇ₐ = TSO.extended_lookup(eos_extended, :∇ₐ, lnrho_local, lnt_local)
        χr = TSO.extended_lookup(eos_extended, :χᵨ, lnrho_local, lnt_local)
        χt = TSO.extended_lookup(eos_extended, :χₜ, lnrho_local, lnt_local)

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
            f_val = 2.0 * U * xi^3 + xi^2 + U * xi - super_adi
            df_dz = 6.0 * U * xi^2 + 2.0 * xi + U
            dxi = f_val / df_dz
            xi -= dxi
            if abs(dxi) < 1e-6 * xi; break; end
        end
        xi = max(xi, 1e-9)

        v_real = v_scale * xi
        # Cap at sound speed
        #if v_real > c_sound
        #    v_real = c_sound
        #    xi = c_sound / v_scale
        #end
        ratio = v_real / c_sound
        soft_factor = (1.0 + ratio^4)^0.25
        v_real = v_real / soft_factor
        xi = xi / soft_factor

        Flux = (0.5 * alpha_mlt) * (exp(lnrho_local) * Cp * T_local) * v_scale * xi^3
        return Flux, v_real
    end

    @inbounds for n in 2:n_depth
        # -- 1. Calculate Gradient (Backward Difference) --
        dlnT = log(T[n] / T[n-1])
        dlnP = log(P_tot[n] / P_tot[n-1])
        ∇_base = dlnT / dlnP
        
        # -- 2. Base Flux --
        F_base, v_base = calc_mlt_local(n, T[n], P_tot[n], ∇_base)
        F_conv[n] = F_base
        v_conv[n] = v_base

        # -- 3. Calculate Derivative dF/dT --
        delta_T = 0.001 * T[n]
        T_pert = T[n] + delta_T
        dlnT_pert = log(T_pert / T[n-1])
        ∇_pert = dlnT_pert / dlnP
        F_pert, _ = calc_mlt_local(n, T_pert, P_tot[n], ∇_pert)
        
        # -- 4. Gustafsson Stability (Eq 20, 21) --
        if F_base <= 1e-10
            b = 0.005 
            T_recipe = T[n] * (1.0 + b)
            
            dlnT_recipe = log(T_recipe / T[n-1])
            ∇_recipe = dlnT_recipe / dlnP
            F_recipe, _ = calc_mlt_local(n, T_recipe, P_tot[n], ∇_recipe)
            
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
end







#= Temperature structure adjustment (non-Feutrier) =#

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



