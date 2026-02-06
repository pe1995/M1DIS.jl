module FeutrierRT

using LinearAlgebra
using SparseArrays

export Atmosphere, solve!, solve_gustafsson!, compute_dT

# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================

mutable struct Atmosphere{T <: AbstractFloat}
    T_eff::T              # Effective Temperature [K]
    z::Vector{T}          # Height (Size: D)
    tau::Vector{T}        # Reference optical depth (Size: D)
    tau_lambda::Matrix{T} # Reference optical depth (Size: Nf, D)
    rho::Vector{T}        # Density [g/cm^3] (Size: D)
    Temp::Vector{T}       # Temperature (Size: D)
    mu::Vector{T}         # Angle cosines (Size: Na)
    w_mu::Vector{T}       # Angle weights (Normalized to sum=1)
    w_lambda::Vector{T}   # Frequency Bin weights (Size: Nf)
    chi::Matrix{T}        # Opacity (Nf x D)
    chi_ref::Vector{T}    # Reference opacity (Size: D)
    B::Matrix{T}          # Planck function (Nf x D)
    dBdT::Matrix{T}       # Derivative of B (Nf x D)
    F_conv::Vector{T}     # Convective Flux (Size: D) 
    dFconv::Vector{T}     # Spatial Derivative dF_conv/dtau (Size: D) 
    dFconv_dT::Vector{T}  # Partial derivative dF_conv/dT (Size: D) 
    eta::Matrix{T}        # Opacity ratio chi / chi_ref
    J_bol::Vector{T}      # Bolometric Mean Intensity (Size: D)
    F_bol::Vector{T}      # Bolometric Flux (Size: D)
    g_rad::Vector{T}      # Radiative Acceleration (Size: D) [cm/s^2] 
    dT::Vector{T}         # Temperature Correction (Size: D) <--- NEW
    J_raw::Array{T, 3}    # Specific Intensity J(mu) (Nf x Na x D)
    Q_heat::Vector{T}     # Heating Rate (Nf x D)
    Q_cool::Vector{T}     # Cooling Rate (Nf x D)
end

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
# 2. INITIALIZATION
# ==============================================================================

"""
    Atmosphere(; T_eff, tau, ...)

Constructor. 
"""
function Atmosphere(; T_eff::T, z::Vector{T}, tau::Vector{T}, rho::Vector{T}, Temp::Vector{T}, 
                    F_conv::Vector{T}, dFconv_dT::Vector{T},
                    mu::Vector{T}, w_mu::Vector{T}, 
                    w_lambda::Vector{T},
                    chi::Matrix{T}, chi_ref::Vector{T}, 
                    B::Matrix{T}, dBdT::Matrix{T}) where T
    D = length(tau)
    Nf = length(w_lambda) 
    Na = length(mu)
    
    # --- 1. Normalize Angle Weights ---
    total_w_mu = sum(w_mu)
    w_mu = (total_w_mu > 0) ? w_mu ./ total_w_mu : w_mu
    
    # --- 2. Eta ---
    eta = zeros(T, Nf, D)
    for d in 1:D
        ref = max(chi_ref[d], 1e-30)
        for f in 1:Nf
            eta[f, d] = chi[f, d] / ref
        end
    end

    # --- 3. Pre-compute Convective Flux Derivative (dF_conv / dtau) ---
    dFconv = zeros(T, D)
    for d in 1:D
        if d == 1
            dt = tau[2] - tau[1]
            dFconv[d] = (F_conv[2] - F_conv[1]) / dt
        elseif d == D
            dt = tau[D] - tau[D-1]
            dFconv[d] = (F_conv[D] - F_conv[D-1]) / dt
        else
            dt = tau[d+1] - tau[d-1]
            dFconv[d] = (F_conv[d+1] - F_conv[d-1]) / dt
        end
    end

    # --- 4. Compute tau_lambda ---
    tau_lambda = zeros(T, Nf, D)
    for f in 1:Nf
        compute_τ!(view(tau_lambda, f, :); z=z, ρκ=chi[f,:])
    end
    
    # --- 4. Allocation ---
    J_raw_init = zeros(T, Nf, Na, D)
    J_bol_init = zeros(T, D)
    F_bol_init = zeros(T, D)
    g_rad_init = zeros(T, D)
    dT_init    = zeros(T, D)
    Q_heat_init = zeros(T, D)
    Q_cool_init = zeros(T, D)

    return Atmosphere{T}(T_eff, deepcopy(z), deepcopy(tau), tau_lambda, deepcopy(rho), deepcopy(Temp), deepcopy(mu), deepcopy(w_mu), deepcopy(w_lambda), 
                         deepcopy(chi), deepcopy(chi_ref), deepcopy(B), deepcopy(dBdT), 
                         deepcopy(F_conv), deepcopy(dFconv), deepcopy(dFconv_dT), 
                         deepcopy(eta), deepcopy(J_bol_init), deepcopy(F_bol_init), deepcopy(g_rad_init), deepcopy(dT_init), deepcopy(J_raw_init), deepcopy(Q_heat_init), deepcopy(Q_cool_init))
end

function Packer(atm::Atmosphere{T}) where T
    Nf = length(atm.w_lambda) 
    Na = length(atm.mu)
    N = Nf * Na
    
    weights = zeros(T, N)
    mu_sq = zeros(T, N)
    freq_idx = zeros(Int, N)
    ang_idx = zeros(Int, N)
    
    idx = 1
    for f in 1:Nf
        for a in 1:Na
            weights[idx]  = atm.w_lambda[f] * atm.w_mu[a]
            mu_sq[idx]    = atm.mu[a]^2
            freq_idx[idx] = f
            ang_idx[idx]  = a
            idx += 1
        end
    end
    return Packer(Nf, Na, N, weights, mu_sq, freq_idx, ang_idx)
end

# ==============================================================================
# 3. SOLVERS
# ==============================================================================

# --- B. GUSTAFSSON (1971) SOLVER ---
"""
    solve_gustafsson!(atm)

Solves the simultaneous system for Radiation (J) and Temperature Correction (dT).
Implements the Gustafsson (1971) integral flux constraint:
F_rad + F_conv = sigma * T_eff^4

Updates atm.J_raw, atm.dT, and derived moments.
"""
function solve_gustafsson!(atm::Atmosphere{T}; include_dT::Bool=true) where T
    D = length(atm.tau)
    pack = Packer(atm)
    N = pack.N_total
    M = include_dT ? N + 1 : N 
    
    sigma_SB = 5.670374419e-5
    F_target = sigma_SB * atm.T_eff^4
    
    rows = Int[]
    cols = Int[]
    vals = T[]
    RHS  = zeros(T, D * M)

    #=b_sum = zeros(T, D)
    db_sum = zeros(T, D)
    for f in 1:size(atm.B, 1)
        db_sum += atm.dBdT[f, :] .*atm.w_lambda[f] .*atm.chi[f,:]
        b_sum += atm.B[f, :] .*atm.w_lambda[f] .*atm.chi[f,:]
    end=#

    
    idx_J(i, d) = (d-1)*M + i
    idx_T(d)    = (d-1)*M + M

    for d in 1:D
        for i in 1:N
            f = pack.freq_idx[i]
            a = pack.ang_idx[i]
            row = idx_J(i, d)

            dt_minus, dt_plus = get_dtau(atm.tau_lambda, f, d)
            
            #eta = atm.eta[f, d]
            B   = atm.B[f, d]
            dB  = atm.dBdT[f, d]
            scale = pack.mu_sq[i] #/ (eta^2)
            
            if d == 1
                # Surface BC
                h = (dt_plus) / sqrt(pack.mu_sq[i])
                diag = 1.0 + h + 0.5*h^2; off = -1.0; src_fac = 0.5*h^2
                
                push!(rows, row); push!(cols, idx_J(i,d)); push!(vals, diag)
                push!(rows, row); push!(cols, idx_J(i,d+1)); push!(vals, off)
                if include_dT; push!(rows, row); push!(cols, idx_T(d)); push!(vals, -src_fac * dB); end
                RHS[row] = src_fac * B
            elseif d == D
                push!(rows, row); push!(cols, idx_J(i,d)); push!(vals, 1.0)
                if include_dT; push!(rows, row); push!(cols, idx_T(d)); push!(vals, -dB); end
                RHS[row] = B 
            else
                denom = dt_minus * dt_plus * (dt_minus + dt_plus) / 2.0
                A = scale / denom * dt_plus
                C = scale / denom * dt_minus
                diag = 1.0 + A + C; src_fac = 1.0
                
                push!(rows, row); push!(cols, idx_J(i,d-1)); push!(vals, -A)
                push!(rows, row); push!(cols, idx_J(i,d)); push!(vals, diag)
                push!(rows, row); push!(cols, idx_J(i,d+1)); push!(vals, -C)
                if include_dT; push!(rows, row); push!(cols, idx_T(d)); push!(vals, -src_fac * dB); end
                RHS[row] = src_fac * B
            end
        end

        if include_dT
            row_flux = idx_T(d)
            if d == 1
                b_sum = 0.0; db_sum = 0.0
                for k in 1:N
                    f = pack.freq_idx[k]
                    a = pack.ang_idx[k]
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
                    
                    w = 4π * atm.w_lambda[f] * atm.w_mu[a] * pack.mu_sq[k]
                    coeff = w / dt_local 

                    push!(rows, row_flux); push!(cols, idx_J(k,d));   push!(vals, coeff)
                    push!(rows, row_flux); push!(cols, idx_J(k,d-1)); push!(vals, -coeff)
                end

                # convective flux
                push!(rows, row_flux); push!(cols, idx_T(d)); push!(vals, atm.dFconv_dT[d]/2)
                push!(rows, row_flux); push!(cols, idx_T(d-1)); push!(vals, atm.dFconv_dT[d-1]/2)

                RHS[row_flux] = F_target - 0.5*(atm.F_conv[d] + atm.F_conv[d-1])
            end
        end
    end
    
    A_mat = sparse(rows, cols, vals, D*M, D*M)
    sol = A_mat \ RHS    
    
    for d in 1:D
        if include_dT
            dT_raw = sol[idx_T(d)]
            atm.dT[d] = dT_raw
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
# 4. POST-PROCESSING
# ==============================================================================

function update_mean_intensity!(atm::Atmosphere{T}) where T
    D = length(atm.tau); Nf = length(atm.w_lambda); Na = length(atm.mu)
    fill!(atm.J_bol, 0.0)
    for d in 1:D; bol_sum = zero(T)
        for f in 1:Nf; sum_J = zero(T)
            for a in 1:Na; sum_J += atm.w_mu[a] * atm.J_raw[f, a, d]; end
            bol_sum += atm.w_lambda[f] * sum_J
        end
        atm.J_bol[d] = bol_sum
    end
end

function compute_flux!(atm::Atmosphere{T}) where T
    D = length(atm.tau); Nf = length(atm.w_lambda); Na = length(atm.mu)
    c_light = 2.99792458e10
    
    fill!(atm.F_bol, 0.0)
    fill!(atm.g_rad, 0.0)
    fill!(atm.Q_heat, 0.0)
    fill!(atm.Q_cool, 0.0)
    
    for f in 1:Nf
        w_lambda_f = atm.w_lambda[f] 
        for a in 1:Na
            ang_factor = 4.0 * pi * atm.w_mu[a] * (atm.mu[a]^2) * w_lambda_f
            for d in 1:D
                if d==1
                    dt = atm.tau_lambda[f, 2] - atm.tau_lambda[f, 1]
                    dJ = atm.J_raw[f,a,2] - atm.J_raw[f,a,1]
                elseif d==D
                    dt = atm.tau_lambda[f, D] - atm.tau_lambda[f, D-1]
                    dJ = atm.J_raw[f,a,D] - atm.J_raw[f,a,D-1]
                else
                    dt = atm.tau_lambda[f, d+1] - atm.tau_lambda[f, d-1]
                    dJ = atm.J_raw[f,a,d+1] - atm.J_raw[f,a,d-1]
                end
                
                flux_term = ang_factor * (dJ / dt)
                
                atm.F_bol[d] += flux_term
                atm.g_rad[d] += flux_term * atm.chi[f, d]
            end
        end
        atm.Q_heat .+= atm.w_lambda[f].*atm.chi[f,:].*atm.J_bol[:]
        atm.Q_cool .+= atm.w_lambda[f].*atm.chi[f,:].*atm.B[f,:]
    end
    
    for d in 1:D
        if atm.rho[d] > 0
            atm.g_rad[d] /= (c_light * atm.rho[d])
        end
    end
end

# ==============================================================================
# 5. HELPERS
# ==============================================================================

function update_solution!(atm, pack, sol)
    N = pack.N_total; D = length(atm.tau)
    for d in 1:D; for k in 1:N
        f = pack.freq_idx[k]; a = pack.ang_idx[k]
        atm.J_raw[f, a, d] = sol[gid(k, d, N)]
    end; end
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
    # Integrate: τ(z) = [ ∫ ρκ dz ]_z0 ^z
    @inbounds for j in eachindex(τ)
        if j==1 
            τ[1] = 0 + abs(z[2] - z[1]) * 0.5 * (ρκ[j])
        else
            τ[j] = τ[j-1] + abs(z[j] - z[j-1]) * 0.5 * (ρκ[j] + ρκ[j-1])
        end
    end
end

end # module