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
            A, B_diag, C, src_fac, ext_fac = feautrier_coeffs(atm, f, d, pack.mu_sq[i])
            
            push!(rows, row); push!(cols, idx_J(i,d)); push!(vals, B_diag)
            if A != 0; push!(rows, row); push!(cols, idx_J(i,d-1)); push!(vals, A); end
            if C != 0; push!(rows, row); push!(cols, idx_J(i,d+1)); push!(vals, C); end
            if include_dT; push!(rows, row); push!(cols, idx_T(d)); push!(vals, -src_fac * dB); end

            RHS[row] = src_fac * B + ext_fac * atm.I_top[f]
        end

        # Flux Constraint
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

function solve_feautrier_1D!(atm::Atmosphere{T}, f::Int, J_out::Matrix{T}, L_acc::AbstractVector{T}) where T
    D, Na = length(atm.tau), length(atm.mu)
    
    @inbounds for a in 1:Na
        rows, cols, vals = Int[], Int[], T[]
        RHS = zeros(T, D)
        mu_sq = atm.mu[a]^2
        
        @inbounds for d in 1:D
            # Get coeffs
            (A, B_diag, C, src_fac, ext_fac) = feautrier_coeffs(atm, f, d, mu_sq)
            
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