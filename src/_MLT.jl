"""
    update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5, Teff=5777.0)

Compute MLT parameters F_conv and g_turb based on MAFAGS implementation.
"""
#=function update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; alpha_mlt=1.5, Teff=5777.0)
    n_depth = length(T)
    F_conv .= 0.0
    v_conv .= 0.0
    g_turb .= 0.0
    dFconv_dT .= 0.0
    P_turb_arr = zeros(n_depth)
    F_target = σ_SB * Teff^4

    lnrho = log.(ρ)
    lnT = log.(T)

    κ_ross = exp.(TSO.extended_lookup(eos_extended, :lnRoss, lnrho, lnT))
    Cp_arr = TSO.extended_lookup(eos_extended, :cₚ, lnrho, lnT)
    Q_arr = TSO.extended_lookup(eos_extended, :Q, lnrho, lnT)
    ∇ₐ_arr = TSO.extended_lookup(eos_extended, :∇ₐ, lnrho, lnT)
    χr_arr = TSO.extended_lookup(eos_extended, :χᵨ, lnrho, lnT)
    χt_arr = TSO.extended_lookup(eos_extended, :χₜ, lnrho, lnT)

    P_rad = (4.0 * σ_SB / (3.0 * c_light)) .* (T .^ 4)
    P_tot = P_gas .+ P_rad
    
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
            Γ₁_approx = χr_arr[n] / (1 - χt_arr[n] * ∇_ad)
            c_sound = sqrt(Γ₁_approx * P_tot[n] / ρ[n])

            Hp = P_tot[n] / (ρ[n] * g_surf)
            Q = Q_arr[n]
            Cp = Cp_arr[n]
            κ = κ_ross[n]

            v_scale = sqrt(g_surf * Q * Hp / 8.0)

            numerator = 24.0 * sqrt(2.0) * σ_SB * T[n]^3
            denominator = κ * ρ[n] * Hp * alpha_mlt * ρ[n] * Cp * v_scale
            U = numerator / denominator

            # Solve cubic equation: 2Uξ³ + ξ² + Uξ - super_adi = 0
            xi = 0.5
            for _ in 1:200
                f_val = 2.0 * U * xi^3 + xi^2 + U * xi - super_adi
                df_dz = 6.0 * U * xi^2 + 2.0 * xi + U
                dxi = f_val / df_dz
                xi -= dxi
                if abs(dxi) < 1e-6 * xi
                    break
                end
            end

            xi = max(xi, 1e-9)
            v_real = v_scale * xi

            # Sound speed limit
            if v_real > c_sound
                v_real = c_sound
                xi = c_sound / v_scale
            end

            # Convective Flux
            #F_conv[n] = (9.0 / 8.0) * (ρ[n] * Cp * T[n]) * v_scale * xi^3
            F_conv[n] = (0.5 * alpha_mlt) * (ρ[n] * Cp * T[n]) * v_scale * xi^3

            df_dxi = 6.0 * U * xi^2 + 2.0 * xi + U
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
        # SAFETY: Cap F_conv at 2.0x the Target Flux.
        # This prevents the solver from seeing the unphysical 10^13 value.
        #=F_max_allowed = 10.0 * F_target 
        if F_conv[n] > F_max_allowed
            # Calculate scaling factor to bring it down to the limit
            scale = F_max_allowed / F_conv[n]
        
            F_conv[n]    *= scale
            dFconv_dT[n] *= scale # CRITICAL: Scale the derivative too!
            v_conv[n]    *= sqrt(scale) # Scale velocity for consistency
        end=#

        #=if τ_ross[n] > 1.0
            min_deriv = 0.01 * (F_target / T[n])
            if dFconv_dT[n] < min_deriv
                dFconv_dT[n] = min_deriv
            end
        end=#
    end

    #=@inbounds for n in 2:n_depth
        #dFconv_dT[n] = (F_conv[n] - F_conv[n-1]) / (T[n] - T[n-1])
        # 2. "Blur" the derivative (Gustafsson Eq 20/21 concept)
        # If F_conv is zero but we are near the convection zone, 
        # provide a small non-zero derivative to "warn" the solver.
        if (abs(F_conv[n]) < 1e-12) && (abs(F_conv[n-1]) > 1e-12)
            # We are above the convection zone.
            # Fake a derivative to prevent overheating.
            dT = (T[n] - T[n-1]) 
            F_conv[n] = F_conv[n-1] - abs(dFconv_dT[n-1]) * dT
            dFconv_dT[n] = (F_conv[n] - F_conv[n-1]) / dT
        end
    end=#
    
    # Enforce Monotonicity of Convective Flux (User Requested)
    # The convective flux should generally increase with depth in the envelope until the bottom
    # Also Cap at F_target to prevents runaway flux
    #=for i in 2:n_depth
        #=if F_conv[i] < F_conv[i-1]
            F_conv[i] = F_conv[i-1]
            dFconv_dT[i] = 0.0
        end=#
        if F_conv[i] > F_target
            F_conv[i] = F_target
        end
    end=#

    for i in 2:n_depth-1
        grad_P_tau = (P_turb_arr[i+1] - P_turb_arr[i-1]) / (τ_ross[i+1] - τ_ross[i-1])
        g_turb[i] = κ_ross[i] * grad_P_tau
    end
    g_turb[1] = g_turb[2]
    g_turb[end] = g_turb[end-1]
end=#

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
    
    κ_ross = exp.(TSO.extended_lookup(eos_extended, :lnRoss, lnrho, lnT))
    Cp_arr = TSO.extended_lookup(eos_extended, :cₚ, lnrho, lnT)
    Q_arr = TSO.extended_lookup(eos_extended, :Q, lnrho, lnT)
    ∇ₐ_arr = TSO.extended_lookup(eos_extended, :∇ₐ, lnrho, lnT)
    χr_arr = TSO.extended_lookup(eos_extended, :χᵨ, lnrho, lnT)
    χt_arr = TSO.extended_lookup(eos_extended, :χₜ, lnrho, lnT)

    P_rad = (4.0 * σ_SB / (3.0 * c_light)) .* (T .^ 4)
    P_tot = P_gas .+ P_rad
    
    # --- HELPER: Local MLT Calculation ---
    # Calculates Flux for a specific T and gradient structure
    function calc_mlt_local(n, T_local, P_local, ∇_local)
        ∇_ad = ∇ₐ_arr[n]
        super_adi = ∇_local - ∇_ad

        if super_adi < 1e-6
            return 0.0, 0.0
        end

        # Local quantities (Assuming rho/Cp/Q don't change drastically with small dT)
        Hp = P_local / (ρ[n] * g_surf)
        Q = Q_arr[n]
        Cp = Cp_arr[n]
        κ = κ_ross[n]
        
        # Optically thick limit approximation for Gamma1
        Γ₁_approx = χr_arr[n] / (1 - χt_arr[n] * ∇_ad)
        c_sound = sqrt(Γ₁_approx * P_local / ρ[n])

        v_scale = sqrt(g_surf * Q * Hp / 8.0)
        
        # U = (24 sqrt(2) sigma T^3) / (kappa rho Hp alpha rho Cp v_scale)
        numerator = 24.0 * sqrt(2.0) * σ_SB * T_local^3
        denominator = κ * ρ[n] * Hp * alpha_mlt * ρ[n] * Cp * v_scale
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
        if v_real > c_sound
            v_real = c_sound
            xi = c_sound / v_scale
        end

        Flux = (0.5 * alpha_mlt) * (ρ[n] * Cp * T_local) * v_scale * xi^3
        return Flux, v_real
    end

    # --- MAIN LOOP ---
    # We skip the very top and bottom to avoid index errors
    @inbounds for n in 2:n_depth
        # 1. Calculate Gradient (Backward Difference)
        # Gustafsson implies coupling between layers. Using T[n] and T[n-1] 
        # ensures that changing T[n] changes the gradient, providing a derivative.
        dlnT = log(T[n] / T[n-1])
        dlnP = log(P_tot[n] / P_tot[n-1])
        ∇_base = dlnT / dlnP
        
        # 2. Base Flux
        F_base, v_base = calc_mlt_local(n, T[n], P_tot[n], ∇_base)
        F_conv[n] = F_base
        v_conv[n] = v_base

        # 3. Calculate Derivative dF/dT
        # Implement "Numerical Differentiation" (Eq 15)
        # We perturb T[n] slightly to see how Flux responds (via T^3 and via Gradient)
        
        # Small perturbation (Standard linearization)
        delta_T = 0.001 * T[n]
        T_pert = T[n] + delta_T
        
        # Recalculate gradient with perturbed T[n] (keeping T[n-1] fixed)
        dlnT_pert = log(T_pert / T[n-1])
        ∇_pert = dlnT_pert / dlnP
        
        F_pert, _ = calc_mlt_local(n, T_pert, P_tot[n], ∇_pert)
        
        # 4. Gustafsson "Recipe" for Stability (Eq 20, 21)
        # "If computed convective flux is zero... we estimate derivatives...
        #  at (T*, T_k+1*) where T* = T(1+b)"
        # This handles the boundary where convection is just turning on.
        
        if F_base <= 1e-10
            b = 0.005 # Recommended value from paper
            T_recipe = T[n] * (1.0 + b)
            
            dlnT_recipe = log(T_recipe / T[n-1])
            ∇_recipe = dlnT_recipe / dlnP
            
            F_recipe, _ = calc_mlt_local(n, T_recipe, P_tot[n], ∇_recipe)
            
            if F_recipe > 1e-10
                # Convection would turn on if we heat this layer!
                # Use this slope to guide the solver.
                dFconv_dT[n] = (F_recipe) / (T_recipe - T[n])
            else
                # Still stable even with perturbation
                dFconv_dT[n] = 0.0
            end
        else
            # Convection is active, use standard numerical derivative
            dFconv_dT[n] = (F_pert - F_base) / delta_T
        end
    end
    
    # Fill edges
    F_conv[1] = F_conv[2]
    dFconv_dT[1] = dFconv_dT[2]
end







#= Temperature structure adjustment =#

deriv(f, x) = begin
    df = diff(f)
    dx = diff(x)

    m = (abs.(df) .< 1e-12) .|| (abs.(dx) .< 1e-12)
    dfdx = df ./ dx
    dfdx[m] .= 0.0

    deriv = similar(f)
    deriv[1] = dfdx[1]
    deriv[2:end] .= dfdx

    deriv[isnan.(deriv) .|| isinf.(deriv)] .= 0.0
    return deriv
end

function update_temperature_correction_mafags!(dT, F_rad, dFrad_dT, F_conv, dFconv_dT, T, Teff; max_step_frac=0.1, min_deriv=1e-12)
    n_depth = length(T)
    F_target = σ_SB * Teff^4
    dT_new = similar(dT)
    F_conv[isnan.(F_conv)] .= 1e-12
    F_rad[isnan.(F_rad)] .= 1e-12
    
    dFconv_dT = max.(deriv(F_conv, T), min_deriv)
    dFrad_dT = max.(deriv(F_rad, T), min_deriv)

    @inbounds for k in 1:n_depth
        Temp = T[k]
        #deriv_rad = 4.0 * σ_SB * Temp^3
        deriv_conv = max(dFconv_dT[k], min_deriv)
        deriv_rad = max(dFrad_dT[k], min_deriv)

        Jacobian = deriv_rad + deriv_conv

        Flux_Error = F_target - (F_rad[k] + F_conv[k])

        step = Flux_Error / Jacobian

        limit = max_step_frac * Temp #min(max_step_frac * Temp, 200.0)
        dT_new[k] = clamp(step, -limit, limit)
    end
    m = (dT_new .* dT) .< 0
    dT_new[m] .*= 0.75
    dT .= dT_new
end

function update_temperature_correction_atlas!(dT, F_rad, F_conv, dFconv_dT, T, τ_grid, Teff; damping=0.5)
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

    local_ratio = abs.(F_tot ./ F_target)
    corr_factor_local = (1.0 ./ local_ratio) .^ 0.25
    #blend = exp.(-0.8 .* τ_grid)
    blend = exp.(τ_grid)
    T_new .= T_new .* (1.0 .+ blend .* (corr_factor_local .- 1.0))
    dT_raw = (T_new .- T)
    dT .= dT_raw 
end 

function update_temperature_correction_feutrier!(dT, F_rad, F_conv, dFconv_dT, T, τ_grid, Teff, lambda_diagonal, J; damping=1.0)
    F_target = σ_SB * Teff^4
    F_tot = F_rad .+ F_conv
    
    # --- 1. Flux Scaling Ratio ---
    # ratio = F_tot / F_target
    ratio = F_tot ./ F_target
    
    # Safety: Clamp scaling to factor of 2 per step (Robustness)
    ratio .= clamp.(ratio, 0.5, 2.0)
    
    # Smooth the ratio (prevents grid noise)
    if length(ratio) > 4
        r_smooth = copy(ratio)
        for i in 2:(length(ratio)-1)
            r_smooth[i] = 0.25*ratio[i-1] + 0.5*ratio[i] + 0.25*ratio[i+1]
        end
        ratio .= r_smooth
    end

    # --- 2. Distort Tau Grid ---
    # Create a new tau grid that *would* have the correct flux
    τ_new = copy(τ_grid)
    τ_new[1] = τ_grid[1] * ratio[1]
    @inbounds for k in 2:length(τ_grid)
        dτ = τ_grid[k] - τ_grid[k-1]
        r_avg = 0.5 * (ratio[k] + ratio[k-1])
        τ_new[k] = τ_new[k-1] + r_avg * dτ
    end
    
    # --- 3. Interpolate Target Temperature ---
    # Find what T *should* be at the current tau
    interp = linear_interpolation(log.(τ_new), log.(T), extrapolation_bc=Line())
    log_T_new = interp(log.(τ_grid))
    T_new = exp.(log_T_new)
    
    dT_radiative = T_new .- T

    # --- 4. Convective Damping (ATLAS Method) ---
    # If Convection is efficient, radiative T changes are less effective.
    # Factor ~ dFrad / (dFrad + dFconv)
    
    dT_arr = zeros(length(T))
    @inbounds for k in 1:length(T)
        Temp = T[k]
        
        deriv_rad = 4.0 * σ_SB * Temp^3
        deriv_conv = max(dFconv_dT[k], 0.0)
        
        # Convective efficiency factor
        # Pure Rad -> 1.0
        # Efficient Conv -> Small
        efficiency = deriv_rad / max(deriv_rad + deriv_conv, 1e-20)
        
        dT_val = dT_radiative[k] * efficiency
        
        # Limiters (Standard ATLAS/MARCS limits)
        limit = min(0.15 * Temp, 500.0) # Generous limit for integral method
        dT[k] = clamp(dT_val, -limit, limit)
    end
    
    # Global Damping
    dT .= damping .* dT
end

"""
    update_temperature_correction_robust!(dT, F_rad, F_conv, dFconv_dT, T, τ_grid, Teff, lambda_diagonal, J; damping=0.5)

A robust temperature correction scheme based on the Unsold-Mawe (Flux-scaling) procedure,
augmented with a small ALI term strictly for the surface layers.
"""
function update_temperature_correction_robust!(dT, F_rad, F_conv, dFconv_dT, T, τ_grid, Teff, lambda_diagonal, J; damping=0.5)
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

function update_temperature_correction!(
    dT, T, Q, dQdT, F_rad, F_conv, Teff;
    α = 1,                 # local RE damping
    β = 1,                 # global flux damping
    max_step_frac = 0.15,
    min_deriv = 1e-30,
    nsurf = 3
)
    N = length(T)
    dT_new = similar(dT)
    @inbounds for i in 1:N
        deriv = max(dQdT[i], min_deriv)
        dT_new[i] = α * Q[i] / deriv
    end

    F_target = σ_SB .* Teff^4
    F_surf   = F_rad .+ F_conv

    εF = (F_surf .- F_target) ./ F_target
    εF = sign.(εF) .* abs.(εF).^0.25
    @inbounds for i in 1:N
        step = dT_new[i] - β * εF[i] * T[i]
        limit = max_step_frac * T[i]
        dT_new[i] = clamp(step, -limit, limit)
    end

    m = (dT_new .* dT) .< 0
    dT_new[m] .*= 0.75

    dT .= dT_new

    return nothing
end



