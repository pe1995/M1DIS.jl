"""
    update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5)

Compute MLT parameters F_conv and g_turb based on MAFAGS implementation.
"""
function update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5)
    n_depth = length(T)
    F_conv .= 0.0
    v_conv .= 0.0
    g_turb .= 0.0
    dFconv_dT .= 0.0
    P_turb_arr = zeros(n_depth)

    lnrho = log.(ρ)
    lnT = log.(T)
    
    # Pre-calculated thermodynamics
    κ_ross = exp.(TSO.extended_lookup(eos_extended, :lnRoss, lnrho, lnT))
    Cp_arr = TSO.extended_lookup(eos_extended, :cₚ, lnrho, lnT)
    Q_arr  = TSO.extended_lookup(eos_extended, :Q, lnrho, lnT)
    ∇ₐ_arr = TSO.extended_lookup(eos_extended, :∇ₐ, lnrho, lnT)
    χr_arr = TSO.extended_lookup(eos_extended, :χᵨ, lnrho, lnT)
    χt_arr = TSO.extended_lookup(eos_extended, :χₜ, lnrho, lnT)
    
    P_rad = (4.0 * σ_SB / (3.0 * c_light)) .* (T.^4)
    P_tot = P_gas .+ P_rad

    # Calculate actual gradient
    ∇_actual = zeros(n_depth)
    @inbounds for n in 2:n_depth-1
        dlnT = log(T[n+1] / T[n-1])
        dlnP = log(P_tot[n+1] / P_tot[n-1])
        ∇_actual[n] = dlnT / dlnP
    end
    ∇_actual[1] = ∇_actual[2]
    ∇_actual[end] = ∇_actual[end-1]

    @inbounds for n in 1:n_depth
        ∇_ad = ∇ₐ_arr[n]
        super_adi = ∇_actual[n] - ∇_ad

        if super_adi > 1e-6
            Γ₁_approx = χr_arr[n] / (1 - χt_arr[n]*∇_ad)
            c_sound = sqrt(Γ₁_approx * P_tot[n] / ρ[n])
            
            Hp = P_tot[n] / (ρ[n] * g_surf)
            Q  = Q_arr[n]
            Cp = Cp_arr[n]
            κ  = κ_ross[n]
            
            v_scale = sqrt(g_surf * Q * Hp / 8.0)
            
            numerator   = 24.0 * sqrt(2.0) * σ_SB * T[n]^3
            denominator = κ * ρ[n] * Hp * alpha_mlt * ρ[n] * Cp * v_scale
            U = numerator / denominator
            
            # Solve cubic equation: 2Uξ³ + ξ² + Uξ - super_adi = 0
            xi = 0.5
            for _ in 1:200
                f_val = 2.0*U*xi^3 + xi^2 + U*xi - super_adi
                df_dz = 6.0*U*xi^2 + 2.0*xi + U
                dxi = f_val / df_dz
                xi -= dxi
                if abs(dxi) < 1e-6 * xi; break; end
            end
            
            xi = max(xi, 1e-9)
            v_real = v_scale * xi
            
            # Sound speed limit
            if v_real > c_sound
                v_real = c_sound
                xi = c_sound / v_scale
            end
            
            # Convective Flux
            F_conv[n] = (9.0/8.0) * (ρ[n] * Cp * T[n]) * v_scale * xi^3

            df_dxi = 6.0*U*xi^2 + 2.0*xi + U  
        	dxi_dGrad = 1.0 / df_dxi
            dF_dGrad = F_conv[n] * (3.0 / xi) * dxi_dGrad
            dGrad_dT = ∇_actual[n] / T[n]
            dFconv_dT[n] = (F_conv[n] / T[n]) + (dF_dGrad * dGrad_dT)

            v_conv[n] = v_real
            P_turb_arr[n] = 0.5 * ρ[n] * v_real^2
        else
            F_conv[n] = 0.0
            v_conv[n] = 0.0
            P_turb_arr[n] = 0.0
            dFconv_dT[n] = 0.0
        end
    end

	#=for n in 2:n_depth-1
    	dFconv_dT[n] = (F_conv[n+1] - F_conv[n-1]) / (T[n+1] - T[n-1])
	end=#
	
    for i in 2:n_depth-1
        grad_P_tau = (P_turb_arr[i+1] - P_turb_arr[i-1]) / (τ_ross[i+1] - τ_ross[i-1])
        g_turb[i] = κ_ross[i] * grad_P_tau
    end
    g_turb[1] = g_turb[2]
    g_turb[end] = g_turb[end-1]
end



function update_mixing_length_mafags!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5, Teff=5777.0)
    # --- MAFAGS Constants ---
    A0 = 1.978929e-04
    Y1 = 0.5
    # ------------------------

    n_depth = length(T)
    
    # Reset outputs
    F_conv .= 0.0
    v_conv .= 0.0
    g_turb .= 0.0
    dFconv_dT .= 0.0
    P_turb_arr = zeros(n_depth)
    
    lnrho = log.(ρ)
    lnT   = log.(T)
    
    # Thermodynamics
    κ_ross = exp.(TSO.extended_lookup(eos_extended, :lnRoss, lnrho, lnT))
    Cp_arr = TSO.extended_lookup(eos_extended, :cₚ, lnrho, lnT)
    Q_arr  = TSO.extended_lookup(eos_extended, :Q, lnrho, lnT)
    ∇ₐ_arr = TSO.extended_lookup(eos_extended, :∇ₐ, lnrho, lnT)
    χt_arr = TSO.extended_lookup(eos_extended, :χₜ, lnrho, lnT)
    χr_arr = TSO.extended_lookup(eos_extended, :χᵨ, lnrho, lnT)

    FT = σ_SB * Teff^4 
    HTOT = FT / (4.0 * π)
    P_rad = (4.0 * σ_SB / (3.0 * c_light)) .* (T.^4)
    P_tot = P_gas .+ P_rad

    # --- 1. Stable Gradient Calculation ---
    # Use Backward Difference (n, n-1) to couple grid points and eliminate spikes.
    GU_arr = zeros(n_depth)
    dlnP_step = zeros(n_depth) 

    @inbounds for n in 2:n_depth
        dlnT = log(T[n] / T[n-1])
        dlnP = log(P_tot[n] / P_tot[n-1])
        dlnP_step[n] = dlnP

        # Safeguard against zero dlnP (prevents division by zero singularities)
        if abs(dlnP) > 1e-12
            GU_arr[n] = dlnT / dlnP
        else
            GU_arr[n] = GU_arr[n-1]
        end
    end
    # Boundary conditions
    GU_arr[1] = GU_arr[2]
    dlnP_step[1] = dlnP_step[2]

    # --- Loop: Depths 1 to N ---
    @inbounds for n in 1:n_depth
        GU   = GU_arr[n]
        GAD  = ∇ₐ_arr[n]
        DELN = GU - GAD
        
        if DELN > 1e-6
            # --- MAFAGS Logic ---
            TAUE = alpha_mlt * P_tot[n] * κ_ross[n] / g_surf
            TAU0 = Y1 * TAUE + 1.0 / TAUE
            Hp   = P_tot[n] / (ρ[n] * g_surf)
            
            # Efficiency factor A
            A_denom = max(GAD * FT, 1e-20) # Safeguard against GAD=0
            A_temp  = alpha_mlt * TAU0 * P_tot[n] * Q_arr[n] / A_denom
            A       = A0 * g_surf * Q_arr[n] * Hp * A_temp^2
            
            # Solve Quadratic: Y^2 + 2Y - 4*A*DELN = 0
            Y = sqrt(1.0 + 4.0 * A * DELN) - 1.0
            
            # Flux & Velocity
            FCN = 0.5 * alpha_mlt * FT * Y^3 / (A * TAU0)
            VC  = (4.0 * π) * FT * Y / (ρ[n] * T[n] * Cp_arr[n] * TAU0)
            
            # --- Sound Speed Limit ---
            Γ₁ = χr_arr[n] / (1.0 - χt_arr[n]*GAD)
            c_sound = sqrt(Γ₁ * P_tot[n] / ρ[n])

            is_limited = false
            if VC > c_sound
                VC = c_sound
                is_limited = true
                # Recalculate Y consistent with VC limit for Flux calc
                Y = VC * (ρ[n] * T[n] * Cp_arr[n] * TAU0) / (4.0 * π * FT)
                FCN = 0.5 * alpha_mlt * FT * Y^3 / (A * TAU0)
            end

            F_conv[n] = FCN
            v_conv[n] = VC
            
            # === ROBUST JACOBIAN (Fixes Convergence) ===
            
            # 1. Gradient Contribution (Dominant Term)
            # This tells the solver that increasing T increases Flux dramatically.
            if is_limited
                dF_dGrad = 0.0 # Flux fixed by thermodynamics
            else
                Y_safe = max(Y, 1e-9)
                # Derived from Y(Y+2) = 4*A*DELN
                dY_dDELN = (2.0 * A) / (Y_safe + 1.0) 
                
                # F ~ Y^3 => dF/dY = 3*F/Y
                dF_dGrad = FCN * (3.0 / Y_safe) * dY_dDELN
            end
            
            # Exact Geometric Derivative: ∂∇/∂T = 1 / (T * dlnP)
            # This is critical. Using GU/T underestimates this by ~1/dlnP (~10-100x).
            # We use abs(dlnP) to ensure the direction is correct (Flux increases with T).
            dGrad_dT = 1.0 / (T[n] * max(abs(dlnP_step[n]), 1e-9))
            
            term_grad = dF_dGrad * dGrad_dT

            # 2. Thermodynamic Contribution (Approximate)
            # We use the safe approximation F/T to avoid numerical noise from grid derivatives.
            term_thermo = FCN / T[n]

            # Sum and ensure positivity to guide solver correctly
            dFconv_dT[n] = term_grad + term_thermo

        else
            F_conv[n] = 0.0
            v_conv[n] = 0.0
            dFconv_dT[n] = 0.0
        end
    end
    
    # Boundary Fallback
    if dFconv_dT[n_depth] == 0.0 && F_conv[n_depth] > 0
         dFconv_dT[n_depth] = 4.0 * F_conv[n_depth] / T[n_depth]
    end

    # Turbulent Pressure Gradient
    P_turb_arr .= 0.5 .* ρ .* v_conv.^2
    @inbounds for i in 2:n_depth-1
        grad_P_tau = (P_turb_arr[i+1] - P_turb_arr[i-1]) / (τ_ross[i+1] - τ_ross[i-1])
        g_turb[i] = κ_ross[i] * grad_P_tau
    end
    g_turb[1] = g_turb[2]; g_turb[end] = g_turb[end-1]
end










#= Temperature structure adjustment =#

function update_temperature_correction_mafags!(dT, F_rad, F_conv, dFconv_dT, T, Teff; max_step_frac=0.08, min_deriv=1e-12)
    n_depth = length(T)
    F_target = σ_SB * Teff^4

    for k in 1:n_depth
        Temp = T[k]

        # Radiation derivative (approx 4F/T)
        deriv_rad = max(4.0 * F_rad[k] / Temp, min_deriv)
        
        # Convection derivative (from updated MLT routine)
        deriv_conv = max(dFconv_dT[k], min_deriv)

        Jacobian = deriv_rad + deriv_conv
        
        Flux_Error = F_target - (F_rad[k] + F_conv[k])
        
        # Calculate raw correction
        dT[k] = Flux_Error / Jacobian

        # Limit step size
        limit = max_step_frac * Temp
        dT[k] = clamp(dT[k], -limit, limit)
    end
end

function update_temperature_correction_atlas!(dT, F_rad, F_conv, T, τ_grid, Teff; damping=0.5)
    F_target = σ_SB * Teff^4
    F_tot = F_rad .+ F_conv
    ratio = (F_tot ./ F_target)
    ratio .= clamp.(ratio, 0.5, 2.0)
	
    τ_new = similar(τ_grid)
    τ_new[1] = τ_grid[1] * ratio[1]
    @inbounds for k in 2:length(τ_grid)
        dτ = τ_grid[k] - τ_grid[k-1]
        r_avg = 0.5 * (ratio[k] + ratio[k-1])
        τ_new[k] = τ_new[k-1] + r_avg * dτ
    end

    interp = linear_interpolation(log.(τ_new), log.(T), extrapolation_bc=Line())
    log_T_new = interp(log.(τ_grid))
    T_new = exp.(log_T_new)
    
    #=surface_flux_ratio = F_tot[1] / F_target
    surf_corr_factor = (1.0 / surface_flux_ratio)^0.25
    
    # Apply the surface offset only where optical depth is small
    blend = exp.(-τ_grid) 
    T_new .= T_new .* (1.0 .+ blend .* (surf_corr_factor - 1.0))
    dT .= damping .* (T_new .- T)=#

    flux_ratio = abs.(F_tot ./ F_target)
    corr_factor = (1.0 ./ flux_ratio) .^0.25
    blend = exp.(-τ_grid.^2) 
    T_new .= T_new .* (1.0 .+ blend .* (corr_factor .- 1.0))

    # apply the damping only to the interior corrections
    dT .= (1 .+ (damping .- 1) .*exp.(-1 ./ τ_grid)) .* (T_new .- T)
end