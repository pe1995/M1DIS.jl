# ==============================================================================
# Full VEF solver with internal VEF Radiative Transfer
# ==============================================================================

function solve_VEF_full!(atm::Atmosphere{T}; include_dT::Bool=true, mode::Symbol=:boundary, tau_trans::Float64=-2.0) where T
    D = length(atm.tau)
    Nf = size(atm.chi, 1)
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4

    fill!(atm.J_bol, 0.0); fill!(atm.F_rad, 0.0)
    fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0); fill!(atm.Q_rad, 0.0)

    schur = zeros(T, D, D)
    RE_res_total = zeros(T, D)

    n_chunks = USE_RT_THREADS[] ? max(1, Threads.nthreads() * 4) : 1
    #n_chunks = 1
    chunk_size = cld(Nf, n_chunks)

    tasks = Vector{Any}(undef, 0)
    sizehint!(tasks, n_chunks)

    for i in 1:n_chunks
        f_start = (i-1)*chunk_size + 1
        f_end   = min(i*chunk_size, Nf)
        if f_start <= f_end
            if USE_RT_THREADS[]
                t = Dagger.@spawn process_frequency_chunk_VEF_full(atm, f_start, f_end, mode, tau_trans)
            else
                t = process_frequency_chunk_VEF_full(atm, f_start, f_end, mode, tau_trans)
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
        for d in 2:D
            schur[d, d] += atm.dFconv_dT[d]
            if d > 1
                schur[d, d-1] += -(atm.Temp[d] / max(atm.Temp[d-1], 1.0)) * atm.dFconv_dT[d]
            end
        end

        RHS = zeros(T, D)
        for d in 1:D
            is_re = use_RE(d, mode, atm, tau_trans)

            if is_re
                RHS[d] = -RE_res_total[d]
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

function process_frequency_chunk_VEF_full(atm::Atmosphere{T}, f_start::Int, f_end::Int, mode::Symbol, tau_trans::Float64) where T
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
    tau_lambda_col = zeros(T, D)
    dchidT_col  = zeros(T, D)

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
    L_nu          = zeros(T, D)

    J_ini_col = zeros(T, D)
    F_ini_col = zeros(T, D)
    K_ini_col = zeros(T, D)

    f_edd_history = zeros(T, D, 4)

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 1000 
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

        # Analytic Stellar Beam Calculation 
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

        fill!(f_edd, 1.0 / 3.0) 
        fill!(J_mean, 0.0)

        h_surf_val = 0.5
        
        for iter in 1:max_scat_iter
            fill!(vef_dl, 0.0); fill!(vef_d,  0.0); fill!(vef_du, 0.0); fill!(vef_rhs, 0.0)
            
            dt1 = max(tau_lambda_col[2] - tau_lambda_col[1], 1e-30)
            
            # surface boundary factor (H/J)
            if iter > 1
                # Compute H/J at the top for the VEF boundary condition.
                # Always use the flux gradient: H = d(fJ)/dτ ≈ Σ wₐ μₐ² (u₂-u₁)/Δτ
                # This is correct for both irradiated and non-irradiated cases
                # because the Feautrier solve is for the *diffuse* field only.
                J_diffuse_top = 0.0
                H_flux_top    = 0.0
                for a in 1:Na
                    J_diffuse_top += atm.w_mu[a] * J_nu[a, 1]
                    #H_flux_top    += atm.w_mu[a] * atm.mu[a]^2 * (J_nu[a, 2] - J_nu[a, 1]) / dt1
                    H_flux_top    += atm.w_mu[a] * atm.mu[a] * J_nu[a, 1]
                end
                h_surf_val = H_flux_top / max(J_diffuse_top, 1e-30)
            end
            
            vef_d[1]   = -2.0*(f_edd[1] + h_surf_val*dt1)/(dt1*dt1) - eps_col[1]
            vef_du[1]  = 2.0*f_edd[2]/(dt1*dt1)
            vef_rhs[1] = -eps_col[1] * B_col[1] - (1.0 - eps_col[1]) * J_ini_col[1]
            
            for d in 2:D-1
                dtm = tau_lambda_col[d]   - tau_lambda_col[d-1]
                dtp = tau_lambda_col[d+1] - tau_lambda_col[d]
                dtc = 0.5*(dtm + dtp)

                vef_dl[d]  = f_edd[d-1] / (dtm * dtc)
                vef_d[d]   = -f_edd[d] * (1.0/dtp + 1.0/dtm) / dtc - eps_col[d]
                vef_du[d]  = f_edd[d+1] / (dtp * dtc)
                vef_rhs[d] = -eps_col[d] * B_col[d] - (1.0 - eps_col[d]) * J_ini_col[d]
            end
            
            vef_d[D]   = 1.0
            vef_rhs[D] = B_col[D]
            
            #solve_tridiagonal_direct!(vef_sol, vef_dl, vef_d, vef_du, vef_rhs)
            M_vef = Tridiagonal(vef_dl[2:end], vef_d, vef_du[1:end-1])
            vef_sol .= M_vef \ vef_rhs

            J_mean .= vef_sol
            # Accelerated Lambda Iteration (ALI) source function update.
            # Uses L_nu (≈ diagonal of Λ operator) from the *previous* iteration's
            # Feautrier solve to accelerate convergence in scattering-dominated layers.
            # Formula: S_new = (ε·B + (1-ε)·(J_total - Λ·S_old)) / (1 - (1-ε)·Λ)
            # On iteration 1, L_nu=0 everywhere, so this reduces to the standard update.
            for d in 1:D
                J_total_d           = J_mean[d] + J_ini_col[d]
                scattering_fraction = 1.0 - eps_col[d]
                ali_numerator       = eps_col[d]*B_col[d] + scattering_fraction*(J_total_d - L_nu[d]*S_col[d])
                ali_denominator     = 1.0 - scattering_fraction*L_nu[d]
                if abs(ali_denominator) > 1e-30
                    S_col[d] = max(ali_numerator / ali_denominator, 0.0)
                else
                    S_col[d] = max(eps_col[d]*B_col[d] + scattering_fraction*J_total_d, 0.0)
                end
            end
            
            fill!(L_nu, 0.0)
            for a in 1:Na
                mu_sq  = atm.mu[a]^2
                weight = atm.w_mu[a]
                
                # Free-streaming top boundary (petitRADTRANS-style).
                # No incoming diffuse radiation from above (beam handled analytically).
                # From I⁺(τ=0)=0: b₁ = 1+2f(1+f), c₁ = -2f², with f = μ/Δτ.
                # Capped at f=1e10 to avoid overflow for very thin top layers.
                mu_val             = sqrt(mu_sq)
                mu_over_dtau_top   = min(mu_val / dt1, 1e10)
                free_stream_diag   = 1.0 + 2.0*mu_over_dtau_top*(1.0 + mu_over_dtau_top)
                free_stream_off    = -2.0*mu_over_dtau_top*mu_over_dtau_top
                tri_d[1]   = free_stream_diag
                tri_du[1]  = free_stream_off
                tri_rhs[1] = S_col[1]
                L_nu[1]   += weight / free_stream_diag

                for d in 2:D-1
                    (A, B, C, src_fac, ext_fac) = feautrier_coeffs(atm, f, d, mu_sq)
                    tri_dl[d]  = A
                    tri_d[d]   = B
                    tri_du[d]  = C
                    tri_rhs[d] = src_fac * S_col[d]
                    L_nu[d] += weight * (src_fac / B)
                end
                
                (A, B, C, src_fac, _) = feautrier_coeffs(atm, f, D, mu_sq)
                tri_dl[D]  = A
                tri_d[D]   = B
                tri_rhs[D] = src_fac * S_col[D]
                L_nu[D] += weight * (src_fac / B)
                
                #solve_tridiagonal_direct!(tri_sol, tri_dl, tri_d, tri_du, tri_rhs)
                M_tri = Tridiagonal(tri_dl[2:end], tri_d, tri_du[1:end-1])
                tri_sol .= M_tri \ tri_rhs
                
                for d in 1:D
                    J_nu[a, d] = tri_sol[d]
                end
            end
            
            max_err = 0.0
            for d in 1:D
                j_sum = 0.0
                k_sum = 0.0
                for a in 1:Na
                    j_sum += atm.w_mu[a] * J_nu[a, d]
                    k_sum += atm.w_mu[a] * atm.mu[a]^2 * J_nu[a, d]
                end
                
                if j_sum > 1e-30
                    f_edd[d] = k_sum / j_sum
                else
                    f_edd[d] = 1.0 / 3.0
                end
                
                err = abs(J_mean[d] - J_old[d]) / max(J_mean[d], 1e-20)
                max_err = max(max_err, err)
            end
            
            # Ng Acceleration on Eddington factors
            for d in 1:D
                f_edd_history[d, 1] = f_edd_history[d, 2]
                f_edd_history[d, 2] = f_edd_history[d, 3]
                f_edd_history[d, 3] = f_edd_history[d, 4]
                f_edd_history[d, 4] = f_edd[d]
            end
            
            if (iter >= 4) && (iter % 4 == 0)
                A11, A12, A22 = 0.0, 0.0, 0.0
                B1, B2 = 0.0, 0.0
                
                for d in 1:D
                    x0 = f_edd_history[d, 1]
                    x1 = f_edd_history[d, 2]
                    x2 = f_edd_history[d, 3]
                    x3 = f_edd_history[d, 4]
                    
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
                        x1 = f_edd_history[d, 2]
                        x2 = f_edd_history[d, 3]
                        x3 = f_edd_history[d, 4]
                        f_extrap = (1.0 - a1 - a2) * x3 + a1 * x2 + a2 * x1
                        f_extrap = clamp(f_extrap, 1e-4, 1.0)
                        f_edd[d] = f_extrap
                        f_edd_history[d, 4] = f_extrap 
                    end
                end
            end

            J_old .= J_mean
            
            if (max_err < tol) || (!do_scattering)
                break 
            end
        end

        # angle-averaged moments
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
            kabs = chi_col[d] - sig_col[d]
            RE_res[d] += w_f * kabs * (J_mean[d] - B_col[d])
            P_rad_part[d] += (4π * w_f / c_light) * K_mean[d]

            if use_RE(d, mode, atm, tau_trans)
                schur_part[d, d] += w_f * dchidT_col[d] * (J_mean[d] - B_col[d])
            end
            
            flux_sum = 0.0
            for a in 1:Na
                ang = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
                dJ, dt = 0.0, 1.0
                if d == 1
                    # Use free-streaming relation consistent with top BC
                    dJ = J_nu[a, 1]
                    dt = atm.mu[a]
                elseif d == D
                    dt = max(tau_lambda_col[D] - tau_lambda_col[D-1], 1e-60)
                    dJ = J_nu[a, D] - J_nu[a, D-1]
                else
                    dt_plus  = max(tau_lambda_col[d+1] - tau_lambda_col[d], 1e-60)
                    dt_minus = max(tau_lambda_col[d] - tau_lambda_col[d-1], 1e-60)
                    
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

        factorize_tridiagonal!(vef_dl, vef_d, vef_du)

        for dp in 1:D
            C_dp = (dp < D) ? eps_col[dp] * dB_col[dp] : dB_col[dp]
            if abs(C_dp) < 1e-30
                continue
            end

            invert_tridiagonal_column!(vef_sol, dp, vef_dl, vef_d, vef_du)

            neg_C = (dp == D) ? C_dp : -C_dp
            neg_C_4pi = 4π * neg_C

            @inbounds @simd for d in 1:D
                f_vef[d] = f_edd[d] * vef_sol[d]
            end

            @inbounds begin
                for d in 1:D
                    is_re = use_RE(d, mode, atm, tau_trans)

                    if is_re
                        kabs_d = chi_col[d] - sig_col[d]
                        dJ_d_dTdp = neg_C * vef_sol[d]
                        diag_dB_dT_d = (dp == d) ? dB_col[d] : 0.0
                        schur_part[d, dp] += kabs_d * (dJ_d_dTdp - diag_dB_dT_d)
                    else
                        if d == 1
                            # Consistently use h_surf_val formula for top boundary Schur
                            grad_p = h_surf_val * vef_sol[1]
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

function solve_tridiagonal_direct!(x::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T}, d::Vector{T}) where T
    n = length(d)
    c_prime = zeros(T, n)
    d_prime = zeros(T, n)

    c_prime[1] = c[1] / b[1]
    d_prime[1] = d[1] / b[1]

    for i in 2:n-1
        m = 1.0 / (b[i] - a[i] * c_prime[i-1])
        c_prime[i] = c[i] * m
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) * m
    end

    m = 1.0 / (b[n] - a[n] * c_prime[n-1])
    d_prime[n] = (d[n] - a[n] * d_prime[n-1]) * m

    x[n] = d_prime[n]
    for i in n-1:-1:1
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    end
end