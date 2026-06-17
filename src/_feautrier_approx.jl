# ==============================================================================
# Dagger-based approximate solver
# ==============================================================================

function solve_approximate!(atm::Atmosphere{T}; include_dT::Bool=true, steepness=15.0, tau_trans=-2.0) where T
    D = length(atm.tau)
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4
    
    #Lambda_star = zeros(T, D)
    RE_res = zeros(T, D)
    RE_jac = zeros(T, D)
    K_rad_diag = zeros(T, D)
    K_rad_prev = zeros(T, D)
    dBdT_bol = zeros(T, D)
    kappa_bol = zeros(T, D)
    compute_formal_sol_dagger!(atm, RE_res, RE_jac, K_rad_diag, K_rad_prev, dBdT_bol, kappa_bol)
    
    if include_dT
        solve_T_correction_approximate_blended!(atm, RE_res, RE_jac, K_rad_diag, K_rad_prev, F_target; steepness=steepness, tau_trans=tau_trans)
    end
end

function compute_formal_sol_dagger!(atm::Atmosphere{T}, RE_res::Vector{T}, RE_jac::Vector{T}, K_rad_diag::Vector{T}, K_rad_prev::Vector{T}, dBdT_bol::Vector{T}, kappa_bol::Vector{T}) where T
    D = length(atm.tau)
    Nf = size(atm.chi, 1)
    
    fill!(atm.J_bol, 0.0); fill!(atm.F_rad, 0.0); fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0); fill!(atm.Q_rad, 0.0)
    fill!(RE_res, 0.0); fill!(RE_jac, 0.0)
    fill!(K_rad_diag, 0.0); fill!(K_rad_prev, 0.0)
    fill!(dBdT_bol, 0.0); fill!(kappa_bol, 0.0)
    
    n_chunks = USE_RT_THREADS[] ? max(1, Threads.nthreads() * 4) : 1
    chunk_size = cld(Nf, n_chunks) 
    
    tasks = Vector{Any}(undef, 0) 
    sizehint!(tasks, n_chunks)

    for i in 1:n_chunks
        f_start = (i-1)*chunk_size + 1
        f_end   = min(i*chunk_size, Nf)
        
        if f_start <= f_end
            if USE_RT_THREADS[]
                t = Dagger.@spawn process_frequency_chunk(atm, f_start, f_end)
            else
                t = process_frequency_chunk(atm, f_start, f_end)
            end
            push!(tasks, t)
        end
    end
    
    for t in tasks
        (J_p, F_p, RE_r, RE_j, K_d, K_p, g_p, P_p, dBdT_p, kappa_p) = if USE_RT_THREADS[]
            fetch(t)::NTuple{10, Vector{T}}
        else
            t::NTuple{10, Vector{T}}
        end
        
        atm.J_bol   .+= J_p
        atm.F_rad   .+= F_p
        RE_res      .+= RE_r
        RE_jac      .+= RE_j
        K_rad_diag  .+= K_d
        K_rad_prev  .+= K_p
        atm.g_rad   .+= g_p
        atm.P_rad   .+= P_p
        atm.Q_rad   .+= 4π .* RE_r
        dBdT_bol    .+= dBdT_p
        kappa_bol   .+= kappa_p
    end

    c_light = 2.99792458e10
    for d in 1:D
        if atm.rho[d] > 0
            atm.g_rad[d] /= (c_light * atm.rho[d])
        end
    end
end

function process_frequency_chunk(atm::Atmosphere{T}, f_start::Int, f_end::Int) where T
    D, Na = length(atm.tau), length(atm.mu)
    
    tri_dl  = zeros(T, D)
    tri_d   = zeros(T, D)
    tri_du  = zeros(T, D)
    tri_rhs = zeros(T, D)
    tri_sol = zeros(T, D)
    
    J_part, F_part = zeros(T, D), zeros(T, D)
    P_rad_part = zeros(T, D)
    g_rad_part = zeros(T, D)
    RE_res, RE_jac = zeros(T, D), zeros(T, D)
    K_rad_diag, K_rad_prev = zeros(T, D), zeros(T, D)

    chi_col = zeros(T, D)
    B_col   = zeros(T, D)
    dB_col  = zeros(T, D)
    dchidT_col = zeros(T, D)
    
    dBdT_bol_part = zeros(T, D)
    kappa_bol_part = zeros(T, D)
    
    J_nu = zeros(T, Na, D)
    L_nu = zeros(T, D)

    J_ini_col = zeros(T, D)
    F_ini_col = zeros(T, D)
    K_ini_col = zeros(T, D)

    sig_col = zeros(T, D) 
    S_col   = zeros(T, D) 
    eps_col = zeros(T, D) 
    J_old   = zeros(T, D) 
    j_sum_new = zeros(T, D) 
    tau_lambda_col = zeros(T, D)
    L_nu = zeros(T, D)
    J_history = zeros(T, D, 4)

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 500
    tol = 1e-5
    mu_star = atm.irrad_mu
    f_redist = 0.25 
    
    @inbounds for f in f_start:f_end
        dchidT_col .= view(atm.dchidT, f, :)
        chi_col .= view(atm.chi, f, :)
        B_col   .= view(atm.B, f, :)
        dB_col  .= view(atm.dBdT, f, :)
        tau_lambda_col .= view(atm.tau_lambda, f, :)

        if do_scattering
            sig_col .= view(atm.chi_scat, f, :)
            eps_col .= 1.0 .- (sig_col ./ chi_col)
        else
            eps_col .= 1.0
            sig_col .= 0.0
        end
        
        J_old .= B_col
        
        # Analytic stellar beam calculation
        if atm.I_top[f] > 0.0
            F_arriving = f_redist * π * atm.I_top[f]
            @inbounds for d in 1:D
                attenuation = exp(-tau_lambda_col[d] / mu_star)
                J_ini_col[d] = (F_arriving / (4.0 * π * mu_star)) * attenuation
                F_ini_col[d] = -F_arriving * attenuation
                K_ini_col[d] = J_ini_col[d] * mu_star^2
            end
        else
            fill!(J_ini_col, zero(T))
            fill!(F_ini_col, zero(T))
            fill!(K_ini_col, zero(T))
        end
        
        # lambda iterations
        lambda_formal_solution!(
            atm, f, max_scat_iter, tol, do_scattering,
            eps_col, B_col, J_old, J_ini_col, S_col,
            J_nu, j_sum_new, L_nu,
            tri_dl, tri_d, tri_du, tri_rhs, tri_sol,
            J_history
        )
        
        w_f = 1 #atm.w_lambda[f]
        
        for d in 1:D
            j_sum = 0.0
            for a in 1:Na
                j_sum += atm.w_mu[a] * J_nu[a, d]
            end
            
            J_part[d] += w_f * (j_sum + J_ini_col[d])
            
            kabs = chi_col[d] - sig_col[d]
            term = w_f * kabs
            RE_res[d] += term * (j_sum + J_ini_col[d] - B_col[d])

            # Accumulate absorption-weighted dB/dT for VEF
            kabs_pos = max(kabs, 0.0)
            dBdT_bol_part[d] += kabs_pos * dB_col[d]
            kappa_bol_part[d] += kabs_pos

            jac_J = term * (L_nu[d] - 1.0) * dB_col[d]
            jac_opacity = w_f * dchidT_col[d] * (j_sum + J_ini_col[d]- B_col[d])
            
            # Prevent jac_opacity from making RE_jac positive or vanishingly small.
            # jac_J is always <= 0. We enforce the sum to be <= 0.1 * jac_J.
            RE_jac[d] += jac_J + min(jac_opacity, -0.9 * jac_J)
            #RE_jac[d] += jac_J
            
            flux_sum = 0.0
            J_sum = 0.0
            k_d_sum  = 0.0
            k_p_sum  = 0.0
            
            for a in 1:Na
                ang = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
                
                if d > 1
                    if d == D
                        dt_local = tau_lambda_col[D] - tau_lambda_col[D-1]
                        diff_coeff = ang / max(dt_local, 1e-20)
                        k_d_sum +=  diff_coeff * dB_col[D]
                        k_p_sum += -diff_coeff * dB_col[D-1]
                    else
                        dt_plus = tau_lambda_col[d+1] - tau_lambda_col[d]
                        dt_minus = tau_lambda_col[d] - tau_lambda_col[d-1]
                        
                        w_plus = dt_minus / (dt_plus + dt_minus)
                        w_minus = dt_plus / (dt_plus + dt_minus)
                        
                        diff_plus = ang / max(dt_plus, 1e-20)
                        diff_minus = ang / max(dt_minus, 1e-20)
                        
                        k_d_sum += w_plus * diff_plus * dB_col[d] + w_minus * diff_minus * dB_col[d]
                        k_p_sum += w_minus * (-diff_minus * dB_col[d-1])
                    end
                end
                
                dJ, dt = 0.0, 1.0
                if d == 1
                    dJ = J_nu[a, 2] - J_nu[a, 1]
                    dt = tau_lambda_col[2] - tau_lambda_col[1]
                elseif d == D
                    dJ = J_nu[a, D] - J_nu[a, D-1]
                    dt = tau_lambda_col[D] - tau_lambda_col[D-1]
                else
                    # 3-point central flux derivative for non-uniform grid
                    dt_plus = tau_lambda_col[d+1] - tau_lambda_col[d]
                    dt_minus = tau_lambda_col[d] - tau_lambda_col[d-1]
                    
                    w_plus = dt_minus / (dt_plus + dt_minus)
                    w_minus = dt_plus / (dt_plus + dt_minus)
                    
                    dJ_plus = (J_nu[a, d+1] - J_nu[a, d]) / dt_plus
                    dJ_minus = (J_nu[a, d] - J_nu[a, d-1]) / dt_minus
                    
                    flux_local = w_plus * dJ_plus + w_minus * dJ_minus
                    dJ = flux_local
                    dt = 1.0
                end
                J_sum += (ang / c_light) * J_nu[a, d]
                flux_sum += ang * (dJ / max(dt, 1e-20))
            end
            P_rad_part[d] += J_sum
            F_part[d]     += flux_sum + F_ini_col[d]
            g_rad_part[d] += (flux_sum + F_ini_col[d]) * chi_col[d]
            K_rad_diag[d] += k_d_sum
            K_rad_prev[d] += k_p_sum
        end
    end
    
    return (J_part, F_part, RE_res, RE_jac, K_rad_diag, K_rad_prev, g_rad_part, P_rad_part, dBdT_bol_part, kappa_bol_part)
end

# ==============================================================================
# Estimate temperature correction for the approximate solver
# ==============================================================================

function solve_T_correction_approximate_blended!(atm::Atmosphere{T}, RE_res::Vector{T}, RE_jac::Vector{T}, K_rad_diag::Vector{T}, K_rad_prev::Vector{T}, F_target::T; steepness=15.0, tau_trans=-2.0) where T
    D = length(atm.tau)
    rows, cols, vals = Int[], Int[], T[]
    RHS = zeros(T, D)
    
    d_conv_top = findfirst(f -> f > 0.01 * F_target, atm.F_conv)
    log_tau_trans = if d_conv_top === nothing || d_conv_top == 1
        tau_trans
    else
        max(tau_trans, min(0.0, log10(atm.tau[d_conv_top]) - 1.0))
    end
    #steepness = 20.0
    
    # Global Scale 
    d_scale = argmin(abs.(log10.(atm.tau) .- 0.0))
    diag_RE_scale = max(-RE_jac[d_scale], 1e-30)
    diag_FC_scale = K_rad_diag[d_scale] + atm.dFconv_dT[d_scale]
    C_scale = diag_RE_scale / (abs(diag_FC_scale) + 1e-30)
        
    @inbounds for d in 1:D
        log_t = log10(atm.tau[d])
        
        # Sigmoid Weighting
        arg = clamp(steepness * (log_t - log_tau_trans), -50.0, 50.0)
        w = 1.0 / (1.0 + exp(arg))
        
        if w < 1e-12
            w = 0.0
        elseif w > (1.0 - 1e-12)
            w = 1.0
        end
        
        # Radiative Equilibrium (RE)
        diag_RE = max(-RE_jac[d], 1e-30)
        rhs_RE  = RE_res[d]

        # Flux Conservation (FC)
        F_curr = atm.F_rad[d] + atm.F_conv[d]
        rhs_FC = F_target - F_curr
        
        val_Rad_d  = K_rad_diag[d]
        val_Conv_d = atm.dFconv_dT[d]
        diag_FC    = val_Rad_d + val_Conv_d
        
        val_Rad_p  = (d > 1) ? K_rad_prev[d] : zero(T)
        val_Conv_p = (d > 1) ? -(atm.Temp[d] / atm.Temp[d-1]) * atm.dFconv_dT[d] : zero(T)
        prev_FC    = val_Rad_p + val_Conv_p
        
        W_RE = w
        W_FC = (1.0 - w) * C_scale 
        
        push!(rows, d); push!(cols, d); push!(vals, W_RE * diag_RE + W_FC * diag_FC)
        
        if d > 1
            push!(rows, d); push!(cols, d-1); push!(vals, W_FC * prev_FC)
        end
        
        RHS[d] = W_RE * rhs_RE + W_FC * rhs_FC
    end
    
    J_mat = sparse(rows, cols, vals, D, D)
    atm.dT .= J_mat \ RHS
end

function solve_T_correction_approximate!(atm::Atmosphere{T}, RE_res::Vector{T}, RE_jac::Vector{T}, K_rad_diag::Vector{T}, K_rad_prev::Vector{T}, F_target::T) where T
    D = length(atm.tau)
    rows, cols, vals = Int[], Int[], T[]
    RHS = zeros(T, D)
        
    @inbounds for d in 1:D
        # Use RE only if convection has safely died (<xx % of F_target) and we are near the surface
        is_pure_rad = (atm.F_conv[d] < 1e-4 * F_target)
        if is_pure_rad && (log10(atm.tau[d]) < -1.0)
            # Use Radiative Equilibrium
            diag_val = -RE_jac[d]
            diag_val = max(diag_val, 1e-30)
            push!(rows, d); push!(cols, d); push!(vals, diag_val)
            RHS[d] = RE_res[d]
        else
            # Flux Conservation
            F_curr = atm.F_rad[d] + atm.F_conv[d]
            RHS[d] = F_target - F_curr
            
            # 1. Convection Terms
            val_Conv_d  = atm.dFconv_dT[d]
            val_Conv_p  = -(atm.Temp[d] / atm.Temp[d-1]) * atm.dFconv_dT[d]

            # 2. Radiative Terms
            val_Rad_d = K_rad_diag[d]
            val_Rad_p = K_rad_prev[d]

            # 3. Fill Matrix (Rad + Conv)
            push!(rows, d); push!(cols, d); push!(vals, val_Rad_d + val_Conv_d)
            
            if d > 1
                push!(rows, d); push!(cols, d-1); push!(vals, val_Rad_p + val_Conv_p)
            end
        end
    end
    
    J_mat = sparse(rows, cols, vals, D, D)
    atm.dT .= J_mat \ RHS
end