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
    ratio = F_tot ./ F_target
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
    dT .= damping .* (T_new .- T);
end