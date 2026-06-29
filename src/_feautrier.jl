# ==============================================================================
# Feautrier RT solver — shared infrastructure
#
# This file contains the core routines shared by all solver variants:
#   - Packer: frequency×angle indexing for the Gustafsson direct solver
#   - lambda_formal_solution!: Feautrier tridiagonal solve with Λ-iteration
#     for scattering and Ng acceleration
#   - feautrier_coeffs: tridiagonal coefficients for the Feautrier equation
#   - Tridiagonal solvers: solve, factorize, invert column
#   - Helpers: optical depth integration, RE/flux mode selection
# ==============================================================================

# ==============================================================================
# Data structures
# ==============================================================================

const USE_RT_THREADS = Ref(true)

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
# Core Lambda iteration
# ==============================================================================

function lambda_formal_solution!(atm::Atmosphere{T}, f::Int, max_scat_iter::Int, tol::Float64, do_scattering::Bool,
                                 eps_col::Vector{T}, B_col::Vector{T}, J_old::Vector{T}, J_ini_col::Vector{T}, S_col::Vector{T},
                                 J_nu::Matrix{T}, j_sum_new::Vector{T}, L_nu::Vector{T},
                                 tri_dl::Vector{T}, tri_d::Vector{T}, tri_du::Vector{T}, tri_rhs::Vector{T}, tri_sol::Vector{T}, J_history::Matrix{T}) where T
    D, Na = length(atm.tau), length(atm.mu)

    for iter in 1:max_scat_iter
        @. S_col = eps_col * B_col + (1.0 - eps_col) * (J_old + J_ini_col)
        
        fill!(L_nu, zero(T))
        fill!(j_sum_new, zero(T))

        for a in 1:Na
            mu_sq  = atm.mu[a]^2
            weight = atm.w_mu[a]
            
            # Bottom Boundary (d = 1)
            (A, B, C, src_fac, _) = feautrier_coeffs(atm, f, 1, mu_sq)
            @inbounds begin
                tri_d[1]   = B
                tri_du[1]  = C
                tri_rhs[1] = src_fac * S_col[1]
                L_nu[1]   += weight * (src_fac / B)
            end

            # Interior 
            for d in 2:D-1
                (A, B, C, src_fac, ext_fac) = feautrier_coeffs(atm, f, d, mu_sq)
                @inbounds begin
                    tri_dl[d]  = A
                    tri_d[d]   = B
                    tri_du[d]  = C
                    tri_rhs[d] = src_fac * S_col[d]
                    L_nu[d]   += weight * (src_fac / B)
                end
            end
            
            # Top Boundary (d = D)
            (A, B, C, src_fac, _) = feautrier_coeffs(atm, f, D, mu_sq)
            @inbounds begin
                tri_dl[D]  = A
                tri_d[D]   = B
                tri_rhs[D] = src_fac * S_col[D]
                L_nu[D]   += weight * (src_fac / B)
            end
            
            solve_tridiagonal!(tri_sol, tri_dl, tri_d, tri_du, tri_rhs)
            @inbounds for d in 1:D
                J_nu[a, d] = tri_sol[d]
            end
        end

        @inbounds for d in 1:D
            for a in 1:Na
                j_sum_new[d] += atm.w_mu[a] * J_nu[a, d]
            end
        end

        # Ng Acceleration
        @inbounds for d in 1:D
            J_history[d, 1] = J_history[d, 2]
            J_history[d, 2] = J_history[d, 3]
            J_history[d, 3] = J_history[d, 4]
            J_history[d, 4] = j_sum_new[d]
        end

        if (iter >= 4) && (iter % 4 == 0)
            A11, A12, A22 = zero(T), zero(T), zero(T)
            B1, B2 = zero(T), zero(T)
            
            @inbounds for d in 1:D
                x0 = J_history[d, 1]
                x1 = J_history[d, 2]
                x2 = J_history[d, 3]
                x3 = J_history[d, 4]
                
                dx1 = x1 - x0
                dx2 = x2 - x1
                dx3 = x3 - x2
                
                d1 = dx3 - dx2
                d2 = dx3 - dx1 
                
                w = 1.0 / max(x3, T(1e-30))
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
            if abs(det) > T(1e-15) * (A11 * A22 + T(1e-30))
                a1 = (A22 * B1 - A12 * B2) / det
                a2 = (A11 * B2 - A12 * B1) / det
                
                @inbounds for d in 1:D
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
        
        max_err = zero(T)
        nan_detected = false
        
        @inbounds for d in 1:D
            if isnan(j_sum_new[d]) || isinf(j_sum_new[d])
                nan_detected = true
                break
            end
            
            err = abs(j_sum_new[d] - J_old[d]) / max(j_sum_new[d], T(1e-20))
            max_err = max(max_err, err)
        end

        if nan_detected
            j_sum_new .= J_old  
            @inbounds for d in 1:D
                j_old_val = J_old[d]
                for a in 1:Na
                    J_nu[a, d] = j_old_val
                end
            end
            break
        end

        J_old .= j_sum_new
        
        if (max_err < tol) || (!do_scattering)
            break 
        end
    end
end

# ==============================================================================
# Core Feautrier kernels
# ==============================================================================

function feautrier_coeffs(atm::Atmosphere{T}, f::Int, d::Int, mu_sq::T) where T
    dt_minus, dt_plus = get_dtau(atm.tau_lambda, f, d)
    D = length(atm.tau)
    if d == 1
        mu = sqrt(mu_sq)
        tau_slab = dt_plus / mu
        eta = atm.chi[f, 1] / atm.chi_ref[1]
        tau_top  = (eta * atm.tau[1]) / mu
        #tau_top = 0.0

        E_slab = (tau_slab < 0.01) ? 1.0 - tau_slab*(1.0 - 0.5*tau_slab) : exp(-tau_slab)
        E_top  = (tau_top < 0.01)  ? 1.0 - tau_top *(1.0 - 0.5*tau_top)  : exp(-tau_top)

        term_top = 2.0 - E_top * (1.0 + E_slab)

        diag = 1.0
        off  = -E_slab
        src  = 0.5 * (1.0 - E_slab) * term_top
        ext  = 0.5 * E_top * (1.0 - E_slab^2)
        (0.0, diag, off, src, ext)
    elseif d == D
        (0.0, 1.0, 0.0, 1.0, 0.0)
    else
        dtm_safe = max(dt_minus, 1e-30)
        dtp_safe = max(dt_plus, 1e-30)

        # Analytically rescaled Feautrier equation to avoid catastrophic
        # cancellation when `1.0 - A - C` is evaluated for extremely small optical depths.
        # Original: A = -2μ^2/(dtm*(dtm+dtp)), C = -2μ^2/(dtp*(dtm+dtp)), diag = 1 - A - C, src = 1.0
        # We rescale the equation by multiplying it by R = (dtm * dtp) / (2μ^2).
        
        R = (dtm_safe * dtp_safe) / (2.0 * mu_sq)
        A = -dtp_safe / (dtm_safe + dtp_safe)
        C = -dtm_safe / (dtm_safe + dtp_safe)
        diag = R + 1.0
        
        (A, diag, C, R, 0.0)
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
        for i in 1:N
            if !isfinite(x[i])
                x[i] = 0.0
            end
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
        # Guard against NaNs or Infs that could propagate to the Schur matrix
        for i in 1:N
            if !isfinite(x[i])
                x[i] = 0.0
            end
        end
    end
end

# ==============================================================================
# Helpers
# ==============================================================================

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
            τ[1] = abs(z[2] - z[1]) * 0.5 * ρκ[j]
        else
            τ[j] = τ[j-1] + abs(z[j] - z[j-1]) * 0.5 * (ρκ[j] + ρκ[j-1])
        end
    end
end

"""
    use_RE(d, mode, atm, tau_trans) → Bool

Determine whether depth point `d` should use the radiative equilibrium (RE)
constraint rather than the flux conservation constraint for the temperature
correction.

Modes:
  - `:RE`       — use RE (k(J-B)=0) everywhere (except bottom boundary)
  - `:FC`       — use FC (F_target=F_rad+F_conv) everywhere
  - `:switch`   — use RE above `tau_trans`, flux conservation below
  - `:boundary` — use RE only at the topmost point (d=1)
"""
@inline function use_RE(d, mode, atm, tau_trans)
    is_re = false
    is_re = if mode == :RE
        true
    elseif mode == :switch
        log10(atm.tau[d]) < tau_trans
    elseif mode == :boundary
        d == 1
    elseif mode == :FC
        false
    else
        error("Specified dT mode is not known. Please select one of the following modes: :RE, :FC, :switch, :boundary. Received $(mode).")
    end

    if d == length(atm.tau)
        is_re = false
    end
    return is_re
end

# ==============================================================================
# Top-level Feutrier solvers
# ==============================================================================

include("_feautrier_gustafsson.jl")
include("_feautrier_approx.jl")
include("_feautrier_vef.jl")
include("_feautrier_vef_full.jl")
#include("_feutrier_mod.jl")