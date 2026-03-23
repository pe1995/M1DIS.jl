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

function solve_approximate!(atm::Atmosphere{T}; include_dT::Bool=true) where T
    D = length(atm.tau)
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4
    
    #Lambda_star = zeros(T, D)
    RE_res = zeros(T, D)
    RE_jac = zeros(T, D)
    K_rad_diag = zeros(T, D)
    K_rad_prev = zeros(T, D)
    compute_formal_sol_dagger!(atm, RE_res, RE_jac, K_rad_diag, K_rad_prev)
    include_dT && solve_T_correction_approximate!(atm, RE_res, RE_jac, K_rad_diag, K_rad_prev, F_target)
end

function compute_formal_sol_dagger!(atm::Atmosphere{T}, RE_res::Vector{T}, RE_jac::Vector{T}, K_rad_diag::Vector{T}, K_rad_prev::Vector{T}) where T
    D = length(atm.tau)
    Nf = size(atm.chi, 1)
    
    fill!(atm.J_bol, 0.0); fill!(atm.F_rad, 0.0); fill!(atm.g_rad, 0.0); fill!(atm.P_rad, 0.0)
    fill!(RE_res, 0.0); fill!(RE_jac, 0.0)
    fill!(K_rad_diag, 0.0); fill!(K_rad_prev, 0.0)
    
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
        (J_p, F_p, RE_r, RE_j, K_d, K_p, g_p, P_p) = fetch(t)::NTuple{8, Vector{T}}
        
        atm.J_bol   .+= J_p
        atm.F_rad   .+= F_p
        RE_res      .+= RE_r
        RE_jac      .+= RE_j
        K_rad_diag  .+= K_d
        K_rad_prev  .+= K_p
        atm.g_rad   .+= g_p
        atm.P_rad   .+= P_p
        atm.Q_rad   .+= 4π .* RE_r
    end

    c_light = 2.99792458e10
    for d in 1:D
        if atm.rho[d] > 0
            atm.g_rad[d] /= (c_light * atm.rho[d])
        end
    end
end

function solve_tridiagonal!(x::Vector{T}, dl::Vector{T}, d::Vector{T}, du::Vector{T}, r::Vector{T}) where T
    N = length(d)
    
    @inbounds begin
        # Forward Elimination
        inv_d1 = 1.0 / d[1]
        du[1] *= inv_d1
        r[1]  *= inv_d1
        
        for i in 2:N
            pivot_inv = 1.0 / (d[i] - dl[i] * du[i-1])
            
            if i < N
                du[i] *= pivot_inv
            end
            
            r[i] = (r[i] - dl[i] * r[i-1]) * pivot_inv
        end
        
        # Back Substitution
        x[N] = r[N]
        for i in N-1:-1:1
            x[i] = r[i] - du[i] * x[i+1]
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
    
    J_nu = zeros(T, Na, D)
    L_nu = zeros(T, D)

    sig_col = zeros(T, D) 
    S_col   = zeros(T, D) 
    eps_col = zeros(T, D) 
    J_old   = zeros(T, D) 
    j_sum_new = zeros(T, D) 

    do_scattering = !isnothing(atm.chi_scat)
    max_scat_iter = !do_scattering ? 1 : 50
    tol = 1e-2
    
    @inbounds for f in f_start:f_end
        
        chi_col .= view(atm.chi, f, :)
        B_col   .= view(atm.B, f, :)
        dB_col  .= view(atm.dBdT, f, :)

        if do_scattering
            sig_col .= view(atm.chi_scat, f, :)
            eps_col .= 1.0 .- (sig_col ./ chi_col)
        else
            eps_col .= 1.0
            sig_col .= 0.0
        end
        
        J_old .= B_col
        
        # lambda iterations
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
        
        w_f = 1 #atm.w_lambda[f]
        
        for d in 1:D
            j_sum = 0.0
            for a in 1:Na
                j_sum += atm.w_mu[a] * J_nu[a, d]
            end
            
            J_part[d] += w_f * j_sum
            
            term = w_f * (chi_col[d] - sig_col[d])
            RE_res[d] += term * (j_sum - B_col[d])
            RE_jac[d] += term * (L_nu[d] - 1.0) * dB_col[d]
            
            flux_sum = 0.0
            J_sum = 0.0
            k_d_sum  = 0.0
            k_p_sum  = 0.0
            
            for a in 1:Na
                ang = 4π * atm.w_mu[a] * atm.mu[a]^2 * w_f
                
                if d > 1
                    if d == D
                        dt_local = atm.tau_lambda[f, D] - atm.tau_lambda[f, D-1]
                        diff_coeff = ang / max(dt_local, 1e-20)
                        k_d_sum +=  diff_coeff * dB_col[D]
                        k_p_sum += -diff_coeff * dB_col[D-1]
                    else
                        dt_plus = atm.tau_lambda[f, d+1] - atm.tau_lambda[f, d]
                        dt_minus = atm.tau_lambda[f, d] - atm.tau_lambda[f, d-1]
                        
                        w_plus = dt_minus / (dt_plus + dt_minus)
                        w_minus = dt_plus / (dt_plus + dt_minus)
                        
                        diff_plus = ang / max(dt_plus, 1e-20)
                        diff_minus = ang / max(dt_minus, 1e-20)
                        
                        k_d_sum += w_plus * diff_plus * dB_col[d] + w_minus * diff_minus * dB_col[d]
                        k_p_sum += w_minus * (-diff_minus * dB_col[d-1])
                        # The plus contribution (d+1) from dJ_plus is ignored in K_rad matrix because it's approximately tri-diagonal but we fold it away
                    end
                end
                
                dJ, dt = 0.0, 1.0
                if d == 1
                    dJ = J_nu[a, 2] - J_nu[a, 1]
                    dt = atm.tau_lambda[f, 2] - atm.tau_lambda[f, 1]
                elseif d == D
                    dJ = J_nu[a, D] - J_nu[a, D-1]
                    dt = atm.tau_lambda[f, D] - atm.tau_lambda[f, D-1]
                else
                    # 3-point central flux derivative for non-uniform grid
                    dt_plus = atm.tau_lambda[f, d+1] - atm.tau_lambda[f, d]
                    dt_minus = atm.tau_lambda[f, d] - atm.tau_lambda[f, d-1]
                    
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
    
    return (J_part, F_part, RE_res, RE_jac, K_rad_diag, K_rad_prev, g_rad_part, P_rad_part)
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
        
        E_slab = (tau_slab < 0.1) ? 1.0 - tau_slab*(1.0 - 0.5*tau_slab) : exp(-tau_slab)
        E_top  = (tau_top < 0.1)  ? 1.0 - tau_top *(1.0 - 0.5*tau_top)  : exp(-tau_top)
        
        term_top = 2.0 - E_top * (1.0 + E_slab)
        
        diag = 1.0
        off  = -E_slab
        src  = 0.5 * (1.0 - E_slab) * term_top
        ext  = 0.5 * E_top * (1.0 - E_slab^2)
        (0.0, diag, off, src, ext)
    elseif d == D # Diffusion BC 
        (0.0, 1.0, 0.0, 1.0, 0.0) 
    else
        denom = 0.5 * dt_minus * dt_plus * (dt_minus + dt_plus)
        A = -(mu_sq / denom) * dt_plus
        C = -(mu_sq / denom) * dt_minus
        diag = 1.0 - A - C 
        (A, diag, C, 1.0, 0.0)
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