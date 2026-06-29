# ==============================================================================
# VEF solver
#
# Solves the radiative transfer equation using the Feautrier method to obtain
# angle-dependent mean intensities J_nu(mu,d), then computes the Eddington
# factor f = K/J. A VEF moment equation is solved
# to obtain the Jacobian for the temperature correction.
# ==============================================================================

function solve_VEF!(atm::Atmosphere{T}; include_dT::Bool=true, mode::Symbol=:boundary, tau_trans::Float64=-2.0) where T
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
                t = Dagger.@spawn process_frequency_chunk_VEF(atm, f_start, f_end, mode, tau_trans)
            else
                t = process_frequency_chunk_VEF(atm, f_start, f_end, mode, tau_trans)
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

    # Temperature correction 
    if include_dT
        RHS = zeros(T, D)
        for d in 1:D
            is_re = use_RE(d, mode, atm, tau_trans)

            if is_re
                RHS[d] = -RE_res_total[d]
            else
                RHS[d] = F_target - atm.F_rad[d] - atm.F_conv[d]
                schur[d, d] += atm.dFconv_dT[d]
                if d > 1
                    schur[d, d-1] += -(atm.Temp[d] / atm.Temp[d-1]) * atm.dFconv_dT[d]
                end
            end
        end

        row = similar(schur[1, :])
        for d in 1:D
            row .= schur[d, :]
            if all(isfinite, row)
                row_scale = maximum(abs.(row)) + 1e-30
                RHS[d] /= row_scale
                schur[d, :] ./= row_scale
            else
                RHS[d] = 0.0
                schur[d, :] .= 0.0
                schur[d, d] = 1.0
            end
        end

        atm.dT .= schur \ RHS
    end
end

function process_frequency_chunk_VEF(atm::Atmosphere{T}, f_start::Int, f_end::Int, mode::Symbol, tau_trans::Float64) where T
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

    chi_col        = zeros(T, D)
    B_col          = zeros(T, D)
    dB_col         = zeros(T, D)
    dchidT_col     = zeros(T, D)
    dchidT_col_scat = zeros(T, D)
    tau_lambda_col = zeros(T, D)
    J_nu           = zeros(T, Na, D)

    sig_col   = zeros(T, D)
    S_col     = zeros(T, D)
    eps_col   = zeros(T, D)
    J_old     = zeros(T, D)
    j_sum_new = zeros(T, D)
    J_history = zeros(T, D, 4)
    L_nu      = zeros(T, D)

    J_mean  = zeros(T, D)
    K_mean  = zeros(T, D)
    f_edd   = zeros(T, D)
    vef_dl  = zeros(T, D)
    vef_d   = zeros(T, D)
    vef_du  = zeros(T, D)
    vef_rhs = zeros(T, D)
    vef_sol = zeros(T, D)

    schur_dt_inv  = zeros(T, D)
    schur_dtp_inv = zeros(T, D)
    schur_dtm_inv = zeros(T, D)
    schur_wp      = zeros(T, D)
    schur_wm      = zeros(T, D)
    f_vef         = zeros(T, D)

    J_ini_col = zeros(T, D)
    F_ini_col = zeros(T, D)
    K_ini_col = zeros(T, D)
    kabs_col  = zeros(T, D)

    is_re_arr = zeros(Bool, D)
    for d in 1:D
        is_re_arr[d] = use_RE(d, mode, atm, tau_trans)
    end

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 500
    
    _chi_scat = do_scattering ? (atm.chi_scat::Matrix{T}) : Matrix{T}(undef, 0, 0)
    _dchidT_scat = do_scattering ? (atm.dchidT_scat::Matrix{T}) : Matrix{T}(undef, 0, 0)
    tol = 1e-6

    mu_star = atm.irrad_mu
    f_redist = 0.25     # day-side redistribution factor (f = 1/4 for full redistribution)
    @inbounds for f in f_start:f_end
        chi_col        .= view(atm.chi, f, :)
        B_col          .= view(atm.B, f, :)
        dB_col         .= view(atm.dBdT, f, :)
        tau_lambda_col .= view(atm.tau_lambda, f, :)
        dchidT_col     .= view(atm.dchidT, f, :)

        if do_scattering
            sig_col .= view(_chi_scat, f, :)
            dchidT_col_scat .= view(_dchidT_scat, f, :)
            @inbounds for d in 1:D
                eps_col[d] = chi_col[d] > 1e-30 ? 1.0 - (sig_col[d] / chi_col[d]) : 1.0
            end
        else
            sig_col .= 0.0
            dchidT_col_scat .= 0.0
            eps_col .= 1.0
        end

        for d in 1:D
            kabs_col[d] = chi_col[d] - sig_col[d]
        end

        J_old .= B_col

        # Analytic stellar beam 
        use_irr_f = atm.I_top[f] > 0.0
        if use_irr_f
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

        # Feautrier formal solution (with scattering)
        lambda_formal_solution!(
            atm, f, max_scat_iter, tol, do_scattering,
            eps_col, B_col, J_old, J_ini_col, S_col,
            J_nu, j_sum_new, L_nu,
            tri_dl, tri_d, tri_du, tri_rhs, tri_sol,
            J_history
        )

        # angle-averaged moments: J, K, H (flux)
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
            f_edd[d]  = k_sum / j_sum

            J_part[d] += w_f * J_mean[d]
            RE_res[d] += w_f * kabs_col[d] * (J_mean[d] - B_col[d])
            P_rad_part[d] += (4π * w_f / c_light) * K_mean[d]

            flux_sum = 0.0
            for a in 1:Na
                ang = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
                dJ, dt = 0.0, 1.0
                if d == 1
                    dt = max(tau_lambda_col[2] - tau_lambda_col[1], 1e-30)
                    dJ = J_nu[a, 2] - J_nu[a, 1]
                    dJ_dt = dJ / dt
                elseif d == D
                    dt = max(tau_lambda_col[D] - tau_lambda_col[D-1], 1e-30)
                    dJ = J_nu[a, D] - J_nu[a, D-1]
                    dJ_dt = dJ / dt
                else
                    dt_plus  = max(tau_lambda_col[d+1] - tau_lambda_col[d], 1e-30)
                    dt_minus = max(tau_lambda_col[d]   - tau_lambda_col[d-1], 1e-30)

                    w_plus  = dt_minus / (dt_plus + dt_minus)
                    w_minus = dt_plus  / (dt_plus + dt_minus)

                    dJ_plus  = (J_nu[a, d+1] - J_nu[a, d]) / dt_plus
                    dJ_minus = (J_nu[a, d] - J_nu[a, d-1]) / dt_minus

                    dJ_dt = w_plus * dJ_plus + w_minus * dJ_minus
                end

                flux_sum += ang * dJ_dt
            end

            F_part[d]     += flux_sum + F_ini_col[d]
            g_rad_part[d] += (flux_sum + F_ini_col[d]) * chi_col[d]
        end

        # Precalculate coefficients 
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

        # Build VEF moment equation for the diffuse field
        fill!(vef_dl, 0.0)
        fill!(vef_d,  0.0)
        fill!(vef_du, 0.0)
        h_surf_val = 0.0

        # Top boundary 
        begin
            dt1 = max(tau_lambda_col[2] - tau_lambda_col[1], 1e-30)

            H_surf    = 0.0
            j_sum_top = 0.0
            for a in 1:Na
                dJ_dt = (J_nu[a, 2] - J_nu[a, 1]) / dt1
                H_surf += atm.w_mu[a] * atm.mu[a]^2 * dJ_dt
                j_sum_top += atm.w_mu[a] * J_nu[a, 1]
            end
            h_surf_val = H_surf / max(j_sum_top, 1e-30)
            vef_d[1]  = -2.0*(f_edd[1] + h_surf_val*dt1)/(dt1*dt1) - eps_col[1]
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

        # Bottom: diffusion BC
        vef_d[D] = 1.0

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
                for d in 1:D
                    is_re = is_re_arr[d]

                    if is_re
                        # RE Jacobian: d(κ_abs·(J-B))/dT
                        dJ_d_dTdp = neg_C * vef_sol[d]
                        diag_dB_dT_d = (dp == d) ? dB_col[d] : 0.0
                        term = kabs_col[d] * (dJ_d_dTdp - diag_dB_dT_d)
                        
                        schur_part[d, dp] += term
                    else
                        # Flux Jacobian: d(4π·H)/dT
                        if d == 1
                            grad_p = (f_vef[2] - f_vef[1]) * schur_dt_inv[1]
                            schur_part[1, dp] += neg_C_4pi * grad_p
                        elseif d == D
                            grad_m = (f_vef[D] - f_vef[D-1]) * schur_dt_inv[D]
                            schur_part[D, dp] += neg_C_4pi * grad_m
                        else
                            grad_p = (f_vef[d+1] - f_vef[d]) * schur_dtp_inv[d]
                            grad_m = (f_vef[d] - f_vef[d-1]) * schur_dtm_inv[d]
                            dfdJ = schur_wp[d] * grad_p + schur_wm[d] * grad_m
                            schur_part[d, dp] += neg_C_4pi * dfdJ
                        end
                    end
                end
            end
        end
    end

    return (J_part, F_part, RE_res, g_rad_part, P_rad_part, schur_part)
end