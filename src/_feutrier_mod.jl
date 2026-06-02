# ==============================================================================
# Full VEF solver
# ==============================================================================

function solve_VEF_mod!(atm::Atmosphere{T}; include_dT::Bool=true) where T
    D = length(atm.tau)
    Nf = size(atm.chi, 1)
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4

    fill!(atm.J_bol, 0.0); fill!(atm.F_rad, 0.0)
    fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0); fill!(atm.Q_rad, 0.0)

    schur = zeros(T, D, D)
    RE_res_total = zeros(T, D)

    n_chunks = USE_RT_THREADS[] ? max(1, Threads.nthreads() * 4) : 1
    chunk_size = cld(Nf, n_chunks)

    tasks = Vector{Any}(undef, 0)
    sizehint!(tasks, n_chunks)

    for i in 1:n_chunks
        f_start = (i-1)*chunk_size + 1
        f_end   = min(i*chunk_size, Nf)
        if f_start <= f_end
            if USE_RT_THREADS[]
                t = Dagger.@spawn process_frequency_chunk_VEF_mod(atm, f_start, f_end)
            else
                t = process_frequency_chunk_VEF_mod(atm, f_start, f_end)
            end
            push!(tasks, t)
        end
    end

    for t in tasks
        (J_p, F_p, RE_r, g_p, P_p, schur_p) = if USE_RT_THREADS[]
            fetch(t)::Tuple{Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Matrix{T}}
        else
            t::Tuple{Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Matrix{T}}
        end

        atm.J_bol .+= J_p
        atm.F_rad .+= F_p
        atm.g_rad .+= g_p
        atm.P_rad .+= P_p
        atm.Q_rad .+= 4π .* RE_r
        RE_res_total .+= RE_r
        schur .+= schur_p
    end

    c_light = 2.99792458e10
    for d in 1:D
        if atm.rho[d] > 0
            atm.g_rad[d] /= (c_light * atm.rho[d])
        end
    end

    if include_dT
        for d in 1:D
            schur[d, d] += atm.dFconv_dT[d]
            if d > 1
                schur[d, d-1] += -(atm.Temp[d] / atm.Temp[d-1]) * atm.dFconv_dT[d]
            end
        end

        RHS = zeros(T, D)
        for d in 1:D
            RHS[d] = F_target - atm.F_rad[d] - atm.F_conv[d]
        end

        atm.dT .= schur \ RHS
    end
end

function process_frequency_chunk_VEF_mod(atm::Atmosphere{T}, f_start::Int, f_end::Int) where T
    D, Na = length(atm.tau), length(atm.mu)
    c_light = 2.99792458e10

    J_part     = zeros(T, D)
    F_part     = zeros(T, D)
    RE_res     = zeros(T, D)
    g_rad_part = zeros(T, D)
    P_rad_part = zeros(T, D)

    schur_part = zeros(T, D, D)

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
    dchidT_col  = zeros(T, D)

    J_mean  = zeros(T, D)
    K_mean  = zeros(T, D)
    f_edd   = zeros(T, D)
    vef_dl  = zeros(T, D)
    vef_d   = zeros(T, D)
    vef_du  = zeros(T, D)
    vef_rhs = zeros(T, D)
    vef_sol = zeros(T, D)
    inv_col = zeros(T, D)

    schur_dt_inv  = zeros(T, D)
    schur_dtp_inv = zeros(T, D)
    schur_dtm_inv = zeros(T, D)
    schur_wp      = zeros(T, D)
    schur_wm      = zeros(T, D)
    f_vef         = zeros(T, D)
    L_nu          = zeros(T, D)

    J_ini_col = zeros(T, D)
    F_ini_col = zeros(T, D)
    K_ini_col = zeros(T, D)

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 500
    tol = 1e-6

    mu_star = atm.irrad_mu
    f_redist = 0.25 
    use_irr = atm.I_top[1] > 0.0

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

        # -------------------------------------------------------
        # Analytic Stellar Beam Calculation 
        # -------------------------------------------------------
        if use_irr
            F_arriving = f_redist * π * atm.I_top[f]
            #F_arriving = f_redist * atm.I_top[f]
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

        # -------------------------------------------------------
        # Feutrier formal solution
        # -------------------------------------------------------
        lambda_formal_solution!(
            atm, f, max_scat_iter, tol, do_scattering,
            eps_col, B_col, J_old, J_ini_col, S_col,
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
            J_mean[d] = j_sum + J_ini_col[d]
            K_mean[d] = k_sum + K_ini_col[d]
            f_edd[d] = k_sum / j_sum
            #f_edd[d] = K_mean[d] / J_mean[d]

            J_part[d] += w_f * J_mean[d]
            kabs = chi_col[d] - sig_col[d]
            RE_res[d] += w_f * kabs * (J_mean[d] - B_col[d])
            P_rad_part[d] += (4π * w_f / c_light) * K_mean[d]

            # add opacity derivative to the Schur matrix
            #if use_irr; schur_part[d, d] += w_f * dchidT_col[d] * (J_mean[d] - B_col[d]); end
            schur_part[d, d] += w_f * dchidT_col[d] * (J_mean[d] - B_col[d])

            flux_sum = 0.0
            for a in 1:Na
                ang = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
                dJ, dt = 0.0, 1.0
                if d == 1
                    dt = tau_lambda_col[2] - tau_lambda_col[1]
                    dJ = J_nu[a, 2] - J_nu[a, 1]
                    
                    # vacuum: 4π * w_mu * mu * J_nu
                    if use_irr
                        dJ = J_nu[a, 1]
                        dt = atm.mu[a]
                    end
                elseif d == D
                    dt = tau_lambda_col[D] - tau_lambda_col[D-1]
                    dJ = J_nu[a, D] - J_nu[a, D-1]
                else
                    dt_plus  = tau_lambda_col[d+1] - tau_lambda_col[d]
                    dt_minus = tau_lambda_col[d] - tau_lambda_col[d-1]
                    
                    w_plus  = dt_minus / (dt_plus + dt_minus)
                    w_minus = dt_plus  / (dt_plus + dt_minus)
                    
                    dJ_plus  = (J_nu[a, d+1] - J_nu[a, d]) / dt_plus
                    dJ_minus = (J_nu[a, d] - J_nu[a, d-1]) / dt_minus
                    
                    dJ = w_plus * dJ_plus + w_minus * dJ_minus
                    dt = 1.0
                end
                flux_sum += ang * (dJ / dt)
            end
            F_part[d] += flux_sum + F_ini_col[d]
            g_rad_part[d] += (flux_sum + F_ini_col[d]) * chi_col[d]
        end

        # Precalculate Schur derivative coefficients
        for d in 1:D
            if d == 1
                schur_dt_inv[1] = 1.0 / (tau_lambda_col[2] - tau_lambda_col[1])
            elseif d == D
                schur_dt_inv[D] = 1.0 / (tau_lambda_col[D] - tau_lambda_col[D-1])
            else
                dtp = tau_lambda_col[d+1] - tau_lambda_col[d]
                dtm = tau_lambda_col[d] - tau_lambda_col[d-1]
                schur_dtp_inv[d] = 1.0 / dtp
                schur_dtm_inv[d] = 1.0 / dtm
                schur_wp[d] = dtm / (dtp + dtm)
                schur_wm[d] = dtp / (dtp + dtm)
            end
        end

        # -------------------------------------------------------
        # Build VEF moment equation
        # -------------------------------------------------------
        fill!(vef_dl, 0.0)
        fill!(vef_d,  0.0)
        fill!(vef_du, 0.0)
        h_surf_val = 0.0

        # Surface: Vacuum for irradiation
        begin
            dt1 = max(tau_lambda_col[2] - tau_lambda_col[1], 1e-30)
            
            H_surf = 0.0
            j_sum_top = 0.0
            for a in 1:Na
                if use_irr
                    H_surf += atm.w_mu[a] * atm.mu[a] * J_nu[a, 1]
                else
                    H_surf += atm.w_mu[a] * atm.mu[a]^2 * (J_nu[a, 2] - J_nu[a, 1]) / dt1
                end
                j_sum_top += atm.w_mu[a] * J_nu[a, 1]
            end            
            h_surf_val = H_surf / j_sum_top
            vef_d[1]  = -2.0*(f_edd[1] + h_surf_val*dt1)/(dt1*dt1) - eps_col[1]
            vef_du[1] = 2.0*f_edd[2]/(dt1*dt1)
        end

        # Interior 
        for d in 2:D-1
            dtm = tau_lambda_col[d]   - tau_lambda_col[d-1]
            dtp = tau_lambda_col[d+1] - tau_lambda_col[d]
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

            invert_tridiagonal_column!(vef_sol, dp, vef_dl, vef_d, vef_du)

            neg_C = (dp == D) ? C_dp : -C_dp
            neg_C_4pi = 4π * neg_C

            @inbounds @simd for d in 1:D
                f_vef[d] = f_edd[d] * vef_sol[d]
            end

            @inbounds begin
                if use_irr
                    dfdJ_1 = h_surf_val * vef_sol[1]
                else
                    dfdJ_1 = (f_vef[2] - f_vef[1]) * schur_dt_inv[1]
                end
                schur_part[1, dp] += neg_C_4pi * dfdJ_1

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