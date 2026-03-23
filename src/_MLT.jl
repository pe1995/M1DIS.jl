# ============================================================================
# MLT computation (1) of convective quantities
# ============================================================================

function vvmlt(a::Float64, b::Float64, c::Float64)
    d = 0.5 / (b * c)
    e = a^2 / (a + d + sqrt(d * (2.0 * a + d)))
    return sqrt(b * e)
end

function calc_mlt_local(T_local, P_local, ∇_local, eos_extended, g_surf, alpha_mlt, P_rad_local, P_turb_local; pbeta=1.0)
    # 1. Use ONLY Gas + Rad pressure for the scale height (MARCS hscale)
    P_gas_rad = max(P_local - P_turb_local, 1e-30)
    lnpgas_local = log(max(P_local - P_rad_local - P_turb_local, 1e-30))
    lnt_local = log(T_local)

    lnrho_local, lnκ_ross, Cp, Q, ∇ₐ = sample(eos_extended, (:lnRho,:lnRoss, :cₚ, :Q, :∇ₐ), lnpgas_local, lnt_local)
    κ_ross = exp(lnκ_ross)
    Hp = P_gas_rad / (exp(lnrho_local) * g_surf)

    super_adi = ∇_local - ∇ₐ
    if super_adi <= 1e-32
        return 0.0, 0.0
    end
    
    #Γ₁_approx = abs(χr / (1 - χt * ∇ₐ))
    #c_sound = sqrt(Γ₁_approx * P_local / exp(lnrho_local))
    v_scale = alpha_mlt * sqrt(g_surf * Q * Hp / 8.0)
    
    numerator = 24.0 * sqrt(2.0) * σ_SB * T_local^3
    denominator = κ_ross * exp(lnrho_local) * Hp * alpha_mlt * exp(lnrho_local) * Cp * v_scale
    U = numerator / denominator

    # Solve cubic for efficiency factor xi
    # 9/8 Uξ³ + ξ² + 2Uξ - (∇ - ∇ad) = 0 (Kippenhahn & Weigert 1990)
    xi = 0.5
    for _ in 1:500
        xi_sq = xi^2
        f_val = 9.0/8.0 * U * xi_sq * xi + xi_sq + 2.0*U * xi - super_adi
        df_dz = 27.0/8.0 * U * xi_sq + 2.0 * xi + 2.0*U
        dxi = f_val / df_dz
        xi -= dxi
        if abs(dxi) < 1e-32 * xi; break; end
    end
    xi = max(xi, 1e-32)

    v_real = v_scale * xi
    v_real = if pbeta > 0.0
        v_max = sqrt(0.75 * P_local / (pbeta * exp(lnrho_local)))
        #(v_real^-8 + v_max^-8)^(-1/8)
        min(v_real, v_max)
    else
        v_real
    end

    Flux = (0.5 * alpha_mlt) * (exp(lnrho_local) * Cp * T_local) * v_scale * xi^3
    
    return Flux, v_real
end

function calc_mlt_local_marcs(T_local, P_local, ∇_local, eos_extended, g_surf, alpha_mlt, P_rad_local, P_turb_local; py=0.076, pny=8.0, pbeta=1.0)
    P_gas_local = max(P_local - P_rad_local - P_turb_local, 1e-30)
    lnpgas_local = log(P_gas_local)
    lnt_local = log(T_local)

    lnrho_local, lnκ_ross, Cp, Q, ∇ₐ = sample(eos_extended, (:lnRho,:lnRoss, :cₚ, :Q, :∇ₐ), lnpgas_local, lnt_local)
    
    super_adi = ∇_local - ∇ₐ
    if super_adi <= 1e-32
        return 0.0, 0.0
    end

    κ_ross = exp(lnκ_ross)
    ρ_local = exp(lnrho_local)
    
    Hp = (P_gas_local + P_rad_local) / (ρ_local * g_surf)
    
    omega = alpha_mlt * Hp * ρ_local * κ_ross
    y_val = py * omega^2
    theta = omega / (1.0 + y_val)
    
    gamma_marcs_abs = (Cp * ρ_local) / (8.0 * σ_SB * T_local^3 * theta)
    
    a_in = super_adi
    b_in = g_surf * Hp * max(Q, 0.0) * alpha_mlt^2 / pny
    c_in = gamma_marcs_abs^2 
    
    v_real = vvmlt(a_in, b_in, c_in)
    v_limited = if pbeta > 0.0
        v_max = sqrt(0.75 * P_local / (pbeta * ρ_local))
        min(v_real, v_max)
    else
        v_real
    end

    gg = gamma_marcs_abs * v_real
    dd = (gg / (1.0 + gg)) * super_adi
    Flux = (0.5 * Cp * ρ_local * alpha_mlt * T_local) * v_real * dd
    
    return Flux, v_limited
end

#="""
    mlt(T, P, ∇, eos, g, alpha_mlt, P_rad, P_turb)

Compute MLT parameters F_conv and v_conv based on Gustafsson et al. (1970).
This method uses the convective efficiency computation from Kippenhahn & Weigert (1990).
"""
mlt(args...; kwargs...) = calc_mlt_local(args...; kwargs...)=#


#="""
    mlt(T, P, ∇, eos, g, alpha_mlt, P_rad, P_turb)

Compute MLT parameters F_conv and v_conv based on Gustafsson et al. (1970).
This method uses the convective efficiency computation from MARCS.
"""
mlt(args...; kwargs...) = calc_mlt_local_marcs(args...; kwargs...)=#


function update_mixing_length!(F_conv, v_conv, P_rad, P_turb, dFconv_dT, T, P_gas, ρ, τ_ross, eos_extended, g_surf; 
    alpha_mlt=1.5, Teff=5777.0, v_mac=0.0, pbeta=1.0)
    
    n_depth = length(T)
    P_turb_prev = copy(P_turb)

    # Reset outputs
    fill!(F_conv, 0.0)
    fill!(v_conv, 0.0)
    fill!(P_turb, 0.0)
    fill!(dFconv_dT, 0.0)
    
    @inbounds for n in 2:n_depth
        P_rad_n = P_rad[n]
        P_tot_n = P_gas[n] + P_rad_n + P_turb_prev[n]
        
        P_rad_nm1 = P_rad[n-1]
        P_tot_nm1 = P_gas[n-1] + P_rad_nm1 + P_turb_prev[n-1]

        # Calculate Gradient 
        dlnP = log(P_tot_n / P_tot_nm1)
        if abs(dlnP) < 1e-32
            continue
        end
        dlnT = log(T[n] / T[n-1])
        ∇_base = dlnT / dlnP
        
        # Base Flux 
        F_base, v_base = mlt(T[n], P_tot_n, ∇_base, eos_extended, g_surf, alpha_mlt, P_rad_n, P_turb_prev[n]; pbeta=pbeta)
        F_conv[n] = F_base
        v_conv[n] = v_base

        # Calculate Derivative dF/dT (Local T_n)
        delta_T = 1e-3 * T[n]
        T_pert = T[n] + delta_T
        ∇_pert = log(T_pert / T[n-1]) / dlnP
        
        F_pert, _ = mlt(T_pert, P_tot_n, ∇_pert, eos_extended, g_surf, alpha_mlt, P_rad_n, P_turb_prev[n]; pbeta=pbeta)
        
        # Stability fix 
        if F_base <= 1e-10
            b = 0.005 
            T_recipe = T[n] * (1.0 + b)
            ∇_recipe = log(T_recipe / T[n-1]) / dlnP
            
            F_recipe, _ = mlt(T_recipe, P_tot_n, ∇_recipe, eos_extended, g_surf, alpha_mlt, P_rad_n, P_turb_prev[n]; pbeta=pbeta)
            
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
    v_conv[1] = v_conv[2]
    
    # Turbulent Pressure
    P_turb .= 0.25 .* ρ .* (pbeta .* v_conv .^ 2 .+ v_mac .^ 2) .+ 0.75 .* P_turb_prev 
end

# ============================================================================
# Stabilizers
# ============================================================================

"""
    fconv_stabilizer!(arr; passes=3)

Stabilizes convective quantities by applying a simple running mean.
"""
function smooth_array!(arr; passes=1)
    n = length(arr)
    if n < 3 return end
    
    tmp = zeros(eltype(arr), n)
    for _ in 1:passes
        tmp[1] = 0.75 * arr[1] + 0.25 * arr[2]
        tmp[n] = 0.75 * arr[n] + 0.25 * arr[n-1]
        @inbounds for i in 2:n-1
            tmp[i] = 0.25 * arr[i-1] + 0.5 * arr[i] + 0.25 * arr[i+1]
        end
        arr .= tmp
    end
end

"""
    gturb_stabilizer!(g_turb, g_surf; max_fraction=0.1, passes=5, relax=0.2)

Stabilisiert g_turb in-place. Nutzt den aktuellen Inhalt von g_turb als 
Zielwert und relaxiert ihn gegen den vorherigen Zustand.
"""
function gturb_stabilizer!(g_turb, g_surf; max_fraction=1.0, passes=5, relax=0.2)
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


