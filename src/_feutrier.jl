# ==============================================================================
# Data structures
# ==============================================================================

struct Packer{T}
    Nf::Int
    Na::Int
    N_total::Int
    weights::Vector{T}
    mu_sq::Vector{T}
    freq_idx::Vector{Int}
    ang_idx::Vector{Int}
end

@inline gid(i, d, N) = (d - 1) * N + i

# ==============================================================================
# Initialization
# ==============================================================================

function Packer(atm::Atmosphere{T}) where T
    Nf = size(atm.chi, 1) 
    Na = length(atm.mu)
    N = Nf * Na
    
    weights = zeros(T, N)
    mu_sq = zeros(T, N)
    freq_idx = zeros(Int, N)
    ang_idx = zeros(Int, N)
    
    idx = 1
    for f in 1:Nf
        for a in 1:Na
            weights[idx]  = atm.w_mu[a]
            mu_sq[idx]    = atm.mu[a]^2
            freq_idx[idx] = f
            ang_idx[idx]  = a
            idx += 1
        end
    end
    return Packer(Nf, Na, N, weights, mu_sq, freq_idx, ang_idx)
end

# ==============================================================================
# Direct solver (Gustafsson)
# ==============================================================================

function solve_gustafsson!(atm::Atmosphere{T}; include_dT::Bool=true) where T
    D = length(atm.tau)
    pack = Packer(atm)
    N = pack.N_total
    M = include_dT ? N + 1 : N 

    # allocate J_raw if not already allocated
    if isnothing(atm.J_raw)
        atm.J_raw = zeros(T, pack.Nf, pack.Na, D)
    end
    fill!(atm.J_raw, 0.0)
    
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4
    
    rows = Int[]
    cols = Int[]
    vals = T[]
    RHS  = zeros(T, D * M)

    idx_J(i, d) = (d-1)*M + i
    idx_T(d)    = (d-1)*M + M

    for d in 1:D
        for i in 1:N
            f = pack.freq_idx[i]
            row = idx_J(i, d)
            
            B   = atm.B[f, d]
            dB  = atm.dBdT[f, d]
            A, B_diag, C, src_fac, ext_fac = feutrier_coeffs(atm, f, d, pack.mu_sq[i])
            
            push!(rows, row); push!(cols, idx_J(i,d)); push!(vals, B_diag)
            if A != 0; push!(rows, row); push!(cols, idx_J(i,d-1)); push!(vals, A); end
            if C != 0; push!(rows, row); push!(cols, idx_J(i,d+1)); push!(vals, C); end
            if include_dT; push!(rows, row); push!(cols, idx_T(d)); push!(vals, -src_fac * dB); end

            RHS[row] = src_fac * B + ext_fac * atm.I_top[f]
        end

        # --- Flux Constraint ---
        if include_dT
            row_flux = idx_T(d)
            if d == 1
                # J = B (opacity-weighted thermal balance) at the surface
                b_sum = 0.0; db_sum = 0.0
                for k in 1:N
                    f = pack.freq_idx[k]
                    term = pack.weights[k] * atm.chi[f, d]
                    push!(rows, row_flux); push!(cols, idx_J(k,d)); push!(vals, term)
                    db_sum += term * atm.dBdT[f, d]
                    b_sum  += term * atm.B[f, d]
                end
                push!(rows, row_flux); push!(cols, idx_T(d)); push!(vals, -db_sum)
                RHS[row_flux] = b_sum
            else
                for k in 1:N
                    f = pack.freq_idx[k]
                    a = pack.ang_idx[k]
                    
                    dt_local = atm.tau_lambda[f, d] - atm.tau_lambda[f, d-1]
                    w = 4π * atm.w_mu[a] * pack.mu_sq[k]
                    coeff = w / dt_local
                    
                    push!(rows, row_flux); push!(cols, idx_J(k,d));   push!(vals, coeff)
                    push!(rows, row_flux); push!(cols, idx_J(k,d-1)); push!(vals, -coeff)
                end
                
                # dFconv/dT: local term at d, plus cross-term from F_conv[d]'s dependence on T[d-1]
                val_Td = 0.5 * atm.dFconv_dT[d]
                push!(rows, row_flux); push!(cols, idx_T(d)); push!(vals, val_Td)
                
                cross_term = -(atm.Temp[d] / atm.Temp[d-1]) * atm.dFconv_dT[d]
                val_Td_minus_1 = 0.5 * (atm.dFconv_dT[d-1] + cross_term)
                push!(rows, row_flux); push!(cols, idx_T(d-1)); push!(vals, val_Td_minus_1)
                
                RHS[row_flux] = F_target - 0.5*(atm.F_conv[d] + atm.F_conv[d-1])
            end
        end
    end
    
    A_mat = sparse(rows, cols, vals, D*M, D*M)
    sol = A_mat \ RHS    
    
    for d in 1:D
        if include_dT
            atm.dT[d] = sol[idx_T(d)]
        end
        for k in 1:N
            J_new = sol[idx_J(k, d)]
            if J_new <= 0.0
                 J_new = 1e-3 * atm.B[pack.freq_idx[k], d] 
            end
            atm.J_raw[pack.freq_idx[k], pack.ang_idx[k], d] = J_new
        end
    end
    
    update_mean_intensity!(atm)
    compute_flux!(atm)
    return nothing
end

# ==============================================================================
# Dagger-based approximate solver
# ==============================================================================

function solve_approximate!(atm::Atmosphere{T}; include_dT::Bool=true, steepness=15.0, tau_trans=-2.0, use_vef::Bool=false) where T
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
        if use_vef
            solve_T_correction_VEF!(atm, RE_res, RE_jac, dBdT_bol, kappa_bol, F_target)
        else
            solve_T_correction_approximate_blended!(atm, RE_res, RE_jac, K_rad_diag, K_rad_prev, F_target; steepness=steepness, tau_trans=tau_trans)
        end
    end
end

function compute_formal_sol_dagger!(atm::Atmosphere{T}, RE_res::Vector{T}, RE_jac::Vector{T}, K_rad_diag::Vector{T}, K_rad_prev::Vector{T}, dBdT_bol::Vector{T}, kappa_bol::Vector{T}) where T
    D = length(atm.tau)
    Nf = size(atm.chi, 1)
    
    fill!(atm.J_bol, 0.0); fill!(atm.F_rad, 0.0); fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0); fill!(atm.Q_rad, 0.0)
    fill!(RE_res, 0.0); fill!(RE_jac, 0.0)
    fill!(K_rad_diag, 0.0); fill!(K_rad_prev, 0.0)
    fill!(dBdT_bol, 0.0); fill!(kappa_bol, 0.0)
    
    n_chunks = max(1, Threads.nthreads() * 4)
    chunk_size = cld(Nf, n_chunks) 
    
    tasks = Vector{Any}(undef, 0) 
    sizehint!(tasks, n_chunks)

    for i in 1:n_chunks
        f_start = (i-1)*chunk_size + 1
        f_end   = min(i*chunk_size, Nf)
        
        if f_start <= f_end
            t = Dagger.@spawn process_frequency_chunk(atm, f_start, f_end)
            push!(tasks, t)
        end
    end
    
    for t in tasks
        (J_p, F_p, RE_r, RE_j, K_d, K_p, g_p, P_p, dBdT_p, kappa_p) = fetch(t)::NTuple{10, Vector{T}}
        
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

    sig_col = zeros(T, D) 
    S_col   = zeros(T, D) 
    eps_col = zeros(T, D) 
    J_old   = zeros(T, D) 
    j_sum_new = zeros(T, D) 
    tau_lambda_col = zeros(T, D)
    L_nu = zeros(T, D)
    J_history = zeros(T, D, 4)

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 100
    tol = 1e-2
    
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
        
        # lambda iterations
        lambda_formal_solution!(
            atm, f, max_scat_iter, tol, do_scattering,
            eps_col, B_col, J_old, S_col,
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
            
            J_part[d] += w_f * j_sum
            
            kabs = chi_col[d] - sig_col[d]
            term = w_f * kabs
            RE_res[d] += term * (j_sum - B_col[d])

            # Accumulate absorption-weighted dB/dT for VEF
            kabs_pos = max(kabs, 0.0)
            dBdT_bol_part[d] += kabs_pos * dB_col[d]
            kappa_bol_part[d] += kabs_pos

            jac_J = term * (L_nu[d] - 1.0) * dB_col[d]
            jac_opacity = w_f * dchidT_col[d] * (j_sum - B_col[d])
            
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
            F_part[d]     += flux_sum
            g_rad_part[d] += flux_sum * chi_col[d]
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
            # --- Flux Conservation ---
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

function solve_T_correction_VEF!(atm::Atmosphere{T}, RE_res::Vector{T}, RE_jac::Vector{T}, dBdT_bol_raw::Vector{T}, kappa_bol_raw::Vector{T}, F_target::T) where T
    D = length(atm.tau)
    c_light = 2.99792458e10

    # ---------------------------------------------------------
    # 1. Compute frequency-integrated Eddington factor f = K / J
    # ---------------------------------------------------------
    # P_rad = (4π/c) * Σ w_mu μ² J_ν  →  K = P_rad * c / (4π)
    # J_bol = Σ w_mu J_ν
    f_edd = zeros(T, D)
    @inbounds for d in 1:D
        K_bol = atm.P_rad[d] * c_light / (4π)
        J_d   = max(atm.J_bol[d], 1e-30)
        f_edd[d] = clamp(K_bol / J_d, 0.05, 0.5)
    end

    # ---------------------------------------------------------
    # 2. Normalize pre-accumulated dB/dT by total absorption opacity
    #    (sums were accumulated in parallel by process_frequency_chunk)
    # ---------------------------------------------------------
    dBdT_bol = zeros(T, D)
    @inbounds for d in 1:D
        dBdT_bol[d] = dBdT_bol_raw[d] / max(kappa_bol_raw[d], 1e-30)
    end

    # ---------------------------------------------------------
    # 3. Build tridiagonal system for δT 
    #
    # The Eddington flux equation:
    #   F_rad(d) ≈ -4π · d(f·J)/dτ_ref  
    #
    # In the diffusion limit J → B, so δF/δT involves d(f·dB/dT·δT)/dτ.
    # We use the reference optical depth τ (Ross) for the derivative.
    #
    # Surface: RE condition (J = B → heating = cooling = 0)
    # Interior: Flux conservation with Eddington factor coupling
    # ---------------------------------------------------------
    dl  = zeros(T, D)  # sub-diagonal
    dd  = zeros(T, D)  # diagonal
    du  = zeros(T, D)  # super-diagonal
    RHS = zeros(T, D)
    dT  = zeros(T, D)

    # --- Surface: use RE ---
    dd[1]  = max(-RE_jac[1], 1e-30)
    RHS[1] = RE_res[1]

    # --- Interior: Eddington-flux conservation ---
    @inbounds for d in 2:D
        F_curr = atm.F_rad[d] + atm.F_conv[d]

        if d < D
            dtau_p = atm.tau[d+1] - atm.tau[d]
            dtau_m = atm.tau[d]   - atm.tau[d-1]

            # Radiative flux Jacobian from finite-difference of 4π·f·dB/dT / dτ
            # Using the same 3-point stencil structure:
            #   δF/δT_{d+1} ≈ 4π · f_{d+1/2} · dBdT_{d+1} / dτ_+
            #   δF/δT_{d}   ≈ -4π · (f_{d+1/2}/dτ_+ + f_{d-1/2}/dτ_-)  · dBdT_d  
            #   δF/δT_{d-1} ≈ 4π · f_{d-1/2} · dBdT_{d-1} / dτ_-
            f_plus  = 0.5 * (f_edd[d] + f_edd[d+1])
            f_minus = 0.5 * (f_edd[d-1] + f_edd[d])

            coeff_p = 4π * f_plus  / max(dtau_p, 1e-20)
            coeff_m = 4π * f_minus / max(dtau_m, 1e-20)

            rad_diag = -(coeff_p + coeff_m) * dBdT_bol[d]
            rad_prev = coeff_m * dBdT_bol[d-1]
            rad_next = coeff_p * dBdT_bol[d+1]

            # Convective flux Jacobian
            conv_diag  = atm.dFconv_dT[d]
            conv_prev  = -(atm.Temp[d] / max(atm.Temp[d-1], 1.0)) * atm.dFconv_dT[d]

            dd[d]  = rad_diag + conv_diag
            dl[d]  = rad_prev + conv_prev
            du[d]  = rad_next
        else
            # Bottom boundary: simple two-point
            dtau_m = atm.tau[D] - atm.tau[D-1]
            f_minus = 0.5 * (f_edd[D-1] + f_edd[D])
            coeff_m = 4π * f_minus / max(dtau_m, 1e-20)

            rad_diag = -coeff_m * dBdT_bol[D]
            rad_prev = coeff_m * dBdT_bol[D-1]

            conv_diag = atm.dFconv_dT[D]
            conv_prev = -(atm.Temp[D] / max(atm.Temp[D-1], 1.0)) * atm.dFconv_dT[D]

            dd[D] = rad_diag + conv_diag
            dl[D] = rad_prev + conv_prev
        end

        RHS[d] = F_target - F_curr
    end

    # Guard against zero diagonal
    @inbounds for d in 1:D
        if abs(dd[d]) < 1e-30
            dd[d] = 1e-30
        end
    end

    # ---------------------------------------------------------
    # 4. Solve tridiagonal system
    # ---------------------------------------------------------
    solve_tridiagonal!(dT, dl, dd, du, RHS)
    atm.dT .= dT
end

# ==============================================================================
# Full per-frequency VEF solver (Rybicki Schur complement)
# ==============================================================================

function solve_VEF!(atm::Atmosphere{T}; include_dT::Bool=true) where T
    D = length(atm.tau)
    Nf = size(atm.chi, 1)
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4

    fill!(atm.J_bol, 0.0); fill!(atm.F_rad, 0.0)
    fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0); fill!(atm.Q_rad, 0.0)

    # Schur complement matrix (D×D) and RHS
    schur = zeros(T, D, D)
    RE_res_total = zeros(T, D)

    n_chunks = max(1, Threads.nthreads() * 4)
    chunk_size = cld(Nf, n_chunks)

    tasks = Vector{Any}(undef, 0)
    sizehint!(tasks, n_chunks)

    for i in 1:n_chunks
        f_start = (i-1)*chunk_size + 1
        f_end   = min(i*chunk_size, Nf)
        if f_start <= f_end
            t = Dagger.@spawn process_frequency_chunk_VEF(atm, f_start, f_end)
            push!(tasks, t)
        end
    end

    for t in tasks
        (J_p, F_p, RE_r, g_p, P_p, schur_p) = fetch(t)::Tuple{Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Matrix{T}}

        atm.J_bol .+= J_p
        atm.F_rad .+= F_p
        atm.g_rad .+= g_p
        atm.P_rad .+= P_p
        atm.Q_rad .+= 4π .* RE_r
        RE_res_total .+= RE_r
        schur     .+= schur_p
    end

    c_light = 2.99792458e10
    for d in 1:D
        if atm.rho[d] > 0
            atm.g_rad[d] /= (c_light * atm.rho[d])
        end
    end

    if include_dT
        for d in 2:D
            schur[d, d] += atm.dFconv_dT[d]
            if d > 1
                schur[d, d-1] += -(atm.Temp[d] / max(atm.Temp[d-1], 1.0)) * atm.dFconv_dT[d]
            end
        end

        RHS = zeros(T, D)
        for d in 1:D
            if d == 1
                RHS[1] = -RE_res_total[1]
            else
                RHS[d] = F_target - atm.F_rad[d] - atm.F_conv[d]
            end
        end

        if abs(schur[1, 1]) < 1e-12
            for j in 1:D
                schur[1, j] = 0.0
            end
            schur[1, 1] = 1.0
            schur[1, 2] = -1.0
            RHS[1] = atm.Temp[2] - atm.Temp[1] 
        end

        atm.dT .= schur \ RHS
    end
end

function process_frequency_chunk_VEF(atm::Atmosphere{T}, f_start::Int, f_end::Int) where T
    D, Na = length(atm.tau), length(atm.mu)
    c_light = 2.99792458e10

    # Physics accumulators
    J_part     = zeros(T, D)
    F_part     = zeros(T, D)
    RE_res     = zeros(T, D)
    g_rad_part = zeros(T, D)
    P_rad_part = zeros(T, D)

    # VEF Schur complement accumulator (D×D)
    schur_part = zeros(T, D, D)

    # Feutrier working arrays
    tri_dl  = zeros(T, D)
    tri_d   = zeros(T, D)
    tri_du  = zeros(T, D)
    tri_rhs = zeros(T, D)
    tri_sol = zeros(T, D)

    chi_col = zeros(T, D)
    B_col   = zeros(T, D)
    dB_col  = zeros(T, D)
    J_nu    = zeros(T, Na, D)

    sig_col   = zeros(T, D)
    S_col     = zeros(T, D)
    eps_col   = zeros(T, D)
    J_old     = zeros(T, D)
    j_sum_new = zeros(T, D)
    tau_lambda_col = zeros(T, D)
    J_history = zeros(T, D, 4)

    # VEF working arrays
    J_mean  = zeros(T, D)
    K_mean  = zeros(T, D)
    f_edd   = zeros(T, D)
    vef_dl  = zeros(T, D)
    vef_d   = zeros(T, D)
    vef_du  = zeros(T, D)
    vef_rhs = zeros(T, D)
    vef_sol = zeros(T, D)
    inv_col = zeros(T, D)
    
    # Pre-calculated derivative coefficients
    schur_dt_inv  = zeros(T, D)
    schur_dtp_inv = zeros(T, D)
    schur_dtm_inv = zeros(T, D)
    schur_wp      = zeros(T, D)
    schur_wm      = zeros(T, D)
    f_vef         = zeros(T, D)
    L_nu          = zeros(T, D)

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 100
    tol = 1e-2

    @inbounds for f in f_start:f_end
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

        # -------------------------------------------------------
        # Feutrier formal solution
        # -------------------------------------------------------
        lambda_formal_solution!(
            atm, f, max_scat_iter, tol, do_scattering,
            eps_col, B_col, J_old, S_col,
            J_nu, j_sum_new, L_nu,
            tri_dl, tri_d, tri_du, tri_rhs, tri_sol,
            J_history
        )

        # -------------------------------------------------------
        # Compute angle-averaged moments: J, K, H (flux)
        # -------------------------------------------------------
        w_f = 1
        fill!(J_mean, 0.0)
        fill!(K_mean, 0.0)

        for d in 1:D
            j_sum = 0.0
            k_sum = 0.0
            for a in 1:Na
                j_sum += atm.w_mu[a] * J_nu[a, d]
                k_sum += atm.w_mu[a] * atm.mu[a]^2 * J_nu[a, d]
            end
            J_mean[d] = j_sum
            K_mean[d] = k_sum

            # Accumulate physics outputs
            J_part[d] += w_f * j_sum
            kabs = chi_col[d] - sig_col[d]
            RE_res[d] += w_f * kabs * (j_sum - B_col[d])
            P_rad_part[d] += (4π * w_f / c_light) * k_sum

            # Flux
            flux_sum = 0.0
            for a in 1:Na
                ang = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
                dJ, dt = 0.0, 1.0
                if d == 1
                    dt = max(tau_lambda_col[2] - tau_lambda_col[1], 1e-60)
                    dJ = J_nu[a, 2] - J_nu[a, 1]
                elseif d == D
                    dt = max(tau_lambda_col[D] - tau_lambda_col[D-1], 1e-60)
                    dJ = J_nu[a, D] - J_nu[a, D-1]
                else
                    dt_plus  = max(tau_lambda_col[d+1] - tau_lambda_col[d], 1e-60)
                    dt_minus = max(tau_lambda_col[d]   - tau_lambda_col[d-1], 1e-60)
                    
                    w_plus  = dt_minus / (dt_plus + dt_minus)
                    w_minus = dt_plus  / (dt_plus + dt_minus)
                    
                    dJ_plus  = (J_nu[a, d+1] - J_nu[a, d]) / dt_plus
                    dJ_minus = (J_nu[a, d] - J_nu[a, d-1]) / dt_minus
                    
                    dJ = w_plus * dJ_plus + w_minus * dJ_minus
                    dt = 1.0
                end
                flux_sum += ang * (dJ / dt)
            end
            F_part[d]     += flux_sum
            g_rad_part[d] += flux_sum * chi_col[d]
        end

        # -------------------------------------------------------
        # Eddington factor f = K / J
        # -------------------------------------------------------
        for d in 1:D
            f_edd[d] = clamp(K_mean[d] / max(J_mean[d], 1e-30), 0.01, 1.0)
        end

        # Precalculate Schur derivative coefficients
        for d in 1:D
            if d == 1
                schur_dt_inv[1] = 1.0 / max(tau_lambda_col[2] - tau_lambda_col[1], 1e-30)
            elseif d == D
                schur_dt_inv[D] = 1.0 / max(tau_lambda_col[D] - tau_lambda_col[D-1], 1e-30)
            else
                dtp = max(tau_lambda_col[d+1] - tau_lambda_col[d], 1e-30)
                dtm = max(tau_lambda_col[d] - tau_lambda_col[d-1], 1e-30)
                schur_dtp_inv[d] = 1.0 / dtp
                schur_dtm_inv[d] = 1.0 / dtm
                schur_wp[d] = dtm / (dtp + dtm)
                schur_wm[d] = dtp / (dtp + dtm)
            end
        end

        # -------------------------------------------------------
        # Build VEF moment equation tridiagonal A_ν
        # -------------------------------------------------------
        fill!(vef_dl, 0.0)
        fill!(vef_d,  0.0)
        fill!(vef_du, 0.0)

        # Surface 
        begin
            dt1 = max(tau_lambda_col[2] - tau_lambda_col[1], 1e-30)
            # Surface Eddington factor h = H/J
            H_surf = 0.0
            for a in 1:Na
                H_surf += atm.w_mu[a] * atm.mu[a]^2 * (J_nu[a, 2] - J_nu[a, 1]) / dt1
            end
            h_surf = H_surf / max(J_mean[1], 1e-30)
            h_surf = clamp(h_surf, 0.0, 2.0)

            vef_d[1]  = -2.0*(f_edd[1] + h_surf*dt1)/(dt1*dt1) - eps_col[1]
            vef_du[1] = 2.0*f_edd[2]/(dt1*dt1)
        end

        # Interior 
        for d in 2:D-1
            dtm = max(tau_lambda_col[d]   - tau_lambda_col[d-1], 1e-30)
            dtp = max(tau_lambda_col[d+1] - tau_lambda_col[d],   1e-30)
            dtc = 0.5*(dtm + dtp)

            vef_dl[d] = f_edd[d-1] / (dtm * dtc)
            vef_d[d]  = -f_edd[d] * (1.0/dtp + 1.0/dtm) / dtc - eps_col[d]
            vef_du[d] = f_edd[d+1] / (dtp * dtc)
        end

        # Bottom: diffusion BC, J=B → δJ = dB/dT δT
        vef_d[D] = 1.0

        # -------------------------------------------------------
        # Schur complement accumulation
        # -------------------------------------------------------
        factorize_tridiagonal!(vef_dl, vef_d, vef_du)

        for dp in 1:D
            C_dp = (dp < D) ? eps_col[dp] * dB_col[dp] : dB_col[dp]
            if abs(C_dp) < 1e-30
                continue
            end

            invert_tridiagonal_column!(vef_sol, dp, vef_dl, vef_d, vef_du)

            neg_C = -C_dp
            neg_C_4pi = 4π * neg_C

            @inbounds @simd for d in 1:D
                f_vef[d] = f_edd[d] * vef_sol[d]
            end

            @inbounds begin
                kabs_1 = chi_col[1] - sig_col[1]
                dJ_1_dTdp = neg_C * vef_sol[1]
                diag_dB_dT = (dp == 1) ? dB_col[1] : 0.0
                
                schur_part[1, dp] += kabs_1 * (dJ_1_dTdp - diag_dB_dT)

                @simd for d in 2:D-1
                    grad_p = (f_vef[d+1] - f_vef[d]) * schur_dtp_inv[d]
                    grad_m = (f_vef[d] - f_vef[d-1]) * schur_dtm_inv[d]
                    dfdJ = schur_wp[d] * grad_p + schur_wm[d] * grad_m
                    schur_part[d, dp] += neg_C_4pi * dfdJ
                end

                dfdJ_D = (f_vef[D] - f_vef[D-1]) * schur_dt_inv[D]
                schur_part[D, dp] += neg_C_4pi * dfdJ_D
            end
        end
    end

    return (J_part, F_part, RE_res, g_rad_part, P_rad_part, schur_part)
end

# ==============================================================================
# Core Lambda iteration
# ==============================================================================

function lambda_formal_solution!(atm::Atmosphere{T}, f::Int, max_scat_iter::Int, tol::Float64, do_scattering::Bool,
                                 eps_col::Vector{T}, B_col::Vector{T}, J_old::Vector{T}, S_col::Vector{T},
                                 J_nu::Matrix{T}, j_sum_new::Vector{T}, L_nu::Vector{T},
                                 tri_dl::Vector{T}, tri_d::Vector{T}, tri_du::Vector{T}, tri_rhs::Vector{T}, tri_sol::Vector{T}, J_history::Matrix{T}) where T
    D, Na = length(atm.tau), length(atm.mu)

    for iter in 1:max_scat_iter
        S_col .= eps_col .* B_col .+ (1.0 .- eps_col) .* J_old
        
        fill!(L_nu, 0.0)
        fill!(j_sum_new, 0.0)

        for a in 1:Na
            mu_sq  = atm.mu[a]^2
            weight = atm.w_mu[a]
            
            (A, B, C, src_fac, ext_fac) = feutrier_coeffs(atm, f, 1, mu_sq)
            tri_d[1]   = B
            tri_du[1]  = C
            tri_rhs[1] = src_fac * S_col[1] + ext_fac * atm.I_top[f]
            
            L_nu[1] += weight * (src_fac / B)

            for d in 2:D-1
                (A, B, C, src_fac, ext_fac) = feutrier_coeffs(atm, f, d, mu_sq)
                tri_dl[d]  = A
                tri_d[d]   = B
                tri_du[d]  = C
                tri_rhs[d] = src_fac * S_col[d]
                
                L_nu[d] += weight * (src_fac / B)
            end
            
            (A, B, C, src_fac, ext_fac) = feutrier_coeffs(atm, f, D, mu_sq)
            tri_dl[D]  = A
            tri_d[D]   = B
            tri_rhs[D] = src_fac * S_col[D]
            
            L_nu[D] += weight * (src_fac / B)
            
            solve_tridiagonal!(tri_sol, tri_dl, tri_d, tri_du, tri_rhs)
            
            for d in 1:D
                J_nu[a, d] = tri_sol[d]
            end
        end

        for d in 1:D
            for a in 1:Na
                j_sum_new[d] += atm.w_mu[a] * J_nu[a, d]
            end
        end

        # --- Ng Acceleration ---
        # Shift history
        for d in 1:D
            J_history[d, 1] = J_history[d, 2]
            J_history[d, 2] = J_history[d, 3]
            J_history[d, 3] = J_history[d, 4]
            J_history[d, 4] = j_sum_new[d]
        end

        if iter >= 4 && iter % 4 == 0
            A11, A12, A22 = 0.0, 0.0, 0.0
            B1, B2 = 0.0, 0.0
            
            for d in 1:D
                x0 = J_history[d, 1]
                x1 = J_history[d, 2]
                x2 = J_history[d, 3]
                x3 = J_history[d, 4]
                
                dx1 = x1 - x0
                dx2 = x2 - x1
                dx3 = x3 - x2
                
                d1 = dx3 - dx2
                d2 = dx2 - dx1
                
                w = 1.0 / max(x3, 1e-30)
                d1_w = d1 * w
                d2_w = d2 * w
                dx3_w = dx3 * w
                
                A11 += d1_w * d1_w
                A12 += d1_w * d2_w
                A22 += d2_w * d2_w
                B1  += dx3_w * d1_w
                B2  += dx3_w * d2_w
            end
            
            det = A11 * A22 - A12 * A12
            if abs(det) > 1e-15 * (A11 * A22 + 1e-30)
                a1 = (A22 * B1 - A12 * B2) / det
                a2 = (A11 * B2 - A12 * B1) / det
                
                for d in 1:D
                    x1 = J_history[d, 2]
                    x2 = J_history[d, 3]
                    x3 = J_history[d, 4]
                    j_extrap = (1.0 - a1 - a2) * x3 + a1 * x2 + a2 * x1
                    if j_extrap > 0.0
                        j_sum_new[d] = j_extrap
                        J_history[d, 4] = j_extrap 
                    end
                end
            end
        end
        # -----------------------
        
        max_err = 0.0
        for d in 1:D
            err = abs(j_sum_new[d] - J_old[d]) / max(j_sum_new[d], 1e-20)
            max_err = max(max_err, err)
        end
        
        J_old .= j_sum_new
        
        if (max_err < tol) || (!do_scattering)
            break 
        end
    end
end

# ==============================================================================
# Core Feutrier kernels
# ==============================================================================

function solve_feutrier_1D!(atm::Atmosphere{T}, f::Int, J_out::Matrix{T}, L_acc::AbstractVector{T}) where T
    D, Na = length(atm.tau), length(atm.mu)
    
    @inbounds for a in 1:Na
        rows, cols, vals = Int[], Int[], T[]
        RHS = zeros(T, D)
        mu_sq = atm.mu[a]^2
        
        @inbounds for d in 1:D
            # Get coeffs
            (A, B_diag, C, src_fac, ext_fac) = feutrier_coeffs(atm, f, d, mu_sq)
            
            if A != 0; push!(rows, d); push!(cols, d-1); push!(vals, A); end
            push!(rows, d); push!(cols, d); push!(vals, B_diag)
            if C != 0; push!(rows, d); push!(cols, d+1); push!(vals, C); end
            
            RHS[d] = src_fac * atm.B[f, d] + ext_fac * atm.I_top[f]
            inv_diag = 1.0 / B_diag
            weight   =  atm.w_mu[a]
            L_acc[d] += weight * (src_fac * inv_diag) 
        end
        
        M = sparse(rows, cols, vals, D, D)
        J_ray = M \ RHS
        
        @inbounds for d in 1:D
            J_out[a, d] = J_ray[d]
        end
    end
end

function feutrier_coeffs(atm::Atmosphere{T}, f::Int, d::Int, mu_sq::T) where T
    dt_minus, dt_plus = get_dtau(atm.tau_lambda, f, d)
    D = length(atm.tau)
    if d == 1 # Estimate infalling radiation
        mu = sqrt(mu_sq)
        tau_slab = dt_plus / mu
        tau_top  = (atm.tau[1] * atm.eta[f, 1]) / mu
        
        E_slab = (tau_slab < 0.01) ? 1.0 - tau_slab*(1.0 - 0.5*tau_slab) : exp(-tau_slab)
        E_top  = (tau_top < 0.01)  ? 1.0 - tau_top *(1.0 - 0.5*tau_top)  : exp(-tau_top)
        
        term_top = 2.0 - E_top * (1.0 + E_slab)
        
        diag = 1.0
        off  = -E_slab
        src  = 0.5 * (1.0 - E_slab) * term_top
        ext  = 0.5 * E_top * (1.0 - E_slab^2)
        (0.0, diag, off, src, ext)
    elseif d == D # Diffusion BC 
        (0.0, 1.0, 0.0, 1.0, 0.0) 
    else
        dtm_safe = max(dt_minus, 1e-30)
        dtp_safe = max(dt_plus, 1e-30)
        
        A = -mu_sq / (0.5 * dtm_safe * (dtm_safe + dtp_safe))
        C = -mu_sq / (0.5 * dtp_safe * (dtm_safe + dtp_safe))
        diag = 1.0 - A - C 
        (A, diag, C, 1.0, 0.0)
    end
end

# ==============================================================================
# Matrix inversion
# ==============================================================================

function solve_tridiagonal!(x::Vector{T}, dl::Vector{T}, d::Vector{T}, du::Vector{T}, r::Vector{T}) where T
    N = length(d)
    @inbounds begin
        piv1 = abs(d[1]) < 1e-60 ? (sign(d[1]) >= 0 ? 1e-60 : -1e-60) : d[1]
        inv_d1 = 1.0 / piv1
        du[1] *= inv_d1
        r[1]  *= inv_d1
        
        for i in 2:N
            pivot = d[i] - dl[i] * du[i-1]
            if abs(pivot) < 1e-60
                pivot = sign(pivot) >= 0 ? 1e-60 : -1e-60
            end
            pivot_inv = 1.0 / pivot
            
            if i < N
                du[i] *= pivot_inv
            end
            r[i] = (r[i] - dl[i] * r[i-1]) * pivot_inv
        end
        
        x[N] = r[N]
        for i in N-1:-1:1
            x[i] = r[i] - du[i] * x[i+1]
        end
    end
end

function factorize_tridiagonal!(dl::Vector{T}, d::Vector{T}, du::Vector{T}) where T
    N = length(d)
    @inbounds begin
        piv1 = abs(d[1]) < 1e-60 ? (sign(d[1]) >= 0 ? 1e-60 : -1e-60) : d[1]
        inv_d1 = 1.0 / piv1
        du[1] *= inv_d1
        d[1] = inv_d1 
        
        for i in 2:N
            pivot = d[i] - dl[i] * du[i-1]
            if abs(pivot) < 1e-60
                pivot = sign(pivot) >= 0 ? 1e-60 : -1e-60
            end
            pivot_inv = 1.0 / pivot
            if i < N
                du[i] *= pivot_inv
            end
            d[i] = pivot_inv 
        end
    end
end

function invert_tridiagonal_column!(x::Vector{T}, dp::Int, dl::Vector{T}, d_inv::Vector{T}, du::Vector{T}) where T
    N = length(d_inv)
    @inbounds begin
        for i in 1:dp-1
            x[i] = 0.0
        end
        x[dp] = d_inv[dp]
        for i in dp+1:N
            x[i] = (-dl[i] * x[i-1]) * d_inv[i]
        end
        for i in N-1:-1:1
            x[i] = x[i] - du[i] * x[i+1]
        end
    end
end

# ==============================================================================
# Helpers
# ==============================================================================

function update_mean_intensity!(atm::Atmosphere{T}) where T
    D = length(atm.tau); Nf = size(atm.chi, 1); Na = length(atm.mu)
    fill!(atm.J_bol, 0.0)
    fill!(atm.Q_rad, 0.0)
    for d in 1:D; bol_sum = zero(T); q_sum = zero(T)
        for f in 1:Nf; sum_J = zero(T)
            for a in 1:Na; sum_J += atm.w_mu[a] * atm.J_raw[f, a, d]; end
            bol_sum += sum_J
            q_sum += atm.chi[f, d] * (sum_J - atm.B[f, d])
        end
        atm.J_bol[d] = bol_sum
        atm.Q_rad[d] = 4π * q_sum
    end
end

function compute_flux!(atm::Atmosphere{T}) where T
    D = length(atm.tau); Nf = size(atm.chi, 1); Na = length(atm.mu)
    c_light = 2.99792458e10
    
    fill!(atm.F_rad, 0.0); fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0)
    
    for f in 1:Nf
        w_f = 1 #atm.w_lambda[f]
        for a in 1:Na
            ang_f = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
            for d in 1:D
                if d==1
                    dt = atm.tau_lambda[f, 2] - atm.tau_lambda[f, 1]
                    dJ = atm.J_raw[f,a,2] - atm.J_raw[f,a,1]
                elseif d==D
                    dt = atm.tau_lambda[f, D] - atm.tau_lambda[f, D-1]
                    dJ = atm.J_raw[f,a,D] - atm.J_raw[f,a,D-1]
                else
                    # 3-point central flux derivative for non-uniform grid
                    dt_plus = atm.tau_lambda[f, d+1] - atm.tau_lambda[f, d]
                    dt_minus = atm.tau_lambda[f, d] - atm.tau_lambda[f, d-1]
                    
                    w_plus = dt_minus / (dt_plus + dt_minus)
                    w_minus = dt_plus / (dt_plus + dt_minus)
                    
                    dJ_plus = (atm.J_raw[f,a,d+1] - atm.J_raw[f,a,d]) / dt_plus
                    dJ_minus = (atm.J_raw[f,a,d] - atm.J_raw[f,a,d-1]) / dt_minus
                    
                    flux_local = w_plus * dJ_plus + w_minus * dJ_minus
                    dJ = flux_local
                    dt = 1.0
                end
                
                flux = ang_f * (dJ / dt)
                atm.F_rad[d] += flux
                atm.g_rad[d] += flux * atm.chi[f, d]
                atm.P_rad[d] += (ang_f / c_light) * atm.J_raw[f, a, d]
            end
        end
    end
    
    for d in 1:D
        if atm.rho[d] > 0
            atm.g_rad[d] /= (c_light * atm.rho[d])
        end
    end
end

function get_dtau(tau, f, d)
    if d == 1
        return (tau[f, 2]-tau[f, 1]), (tau[f, 2]-tau[f, 1])
    elseif d == size(tau, 2)
        return (tau[f, d]-tau[f, d-1]), (tau[f, d]-tau[f, d-1])
    else
        return (tau[f, d]-tau[f, d-1]), (tau[f, d+1]-tau[f, d])
    end
end

function compute_τ!(τ; z, ρκ)
    @inbounds for j in eachindex(τ)
        if j==1 
            τ[1] = 0 + abs(z[2] - z[1]) * 0.5 * (ρκ[j])
        else
            τ[j] = τ[j-1] + abs(z[j] - z[j-1]) * 0.5 * (ρκ[j] + ρκ[j-1])
        end
    end
end