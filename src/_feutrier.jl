function solve_feautrier!(u, A, B, C, RHS, dl, d, du)
    @inbounds begin
        dl .= -A[2:end]
        du .= -C[1:end-1]
        d  .= B
    end
    u .= Tridiagonal(dl, d, du) \ RHS
end

function build_radiative_jacobian!(
    Jmat,               # (Ndepth,Ndepth)
    T, ρ, z,
    eos, opa_extended;
    μ_angles, μ_weights,
    λ_weights = nothing,
)
    N = length(T)
    fill!(Jmat, 0.0)
    opa = opa_extended.opa

    lnrho = log.(ρ)
    lnt   = log.(T)

    nbin = length(opa.λ)
    bwts = isnothing(λ_weights) ? ones(nbin) : λ_weights

    # Workspace
    τ   = zeros(N)
    Δτ  = zeros(N-1)
    A   = zeros(N)
    B   = zeros(N)
    C   = zeros(N)
    RHS = zeros(N)
    u   = zeros(N)
    δu  = zeros(N)
    dl  = zeros(N-1)
    du  = zeros(N-1)
    d   = zeros(N)

    for (bin, bw) in enumerate(bwts)
        S      = lookup(eos, opa, :src, lnrho, lnt, bin)
        dS_dT = TSO.extended_lookup(eos, opa_extended, :dS_dT, lnrho, lnt, bin)
        κρ     = lookup(eos, opa, :κ, lnrho, lnt, bin)

        compute_τ_grid!(τ; z=z, ρκ=κρ)
        Δτ .= diff(τ)

        grad_S = (S[end] - S[end-1]) / Δτ[end]

        for (μ, wμ) in zip(μ_angles, μ_weights)
            μ2 = μ^2
            @inbounds for k = 2:N-1
                dτm = Δτ[k-1]
                dτp = Δτ[k]
                fac = 2μ2 / (dτm + dτp)
                A[k] = fac / dτm
                C[k] = fac / dτp
                B[k] = 1 + fac * (1/dτm + 1/dτp)
            end

            dτ = Δτ[1]
            B[1] = 1 + 2μ / dτ
            C[1] = -2μ / dτ

            dτb = Δτ[end]
            A[end] = -2μ / dτb
            B[end] = 1 + 2μ / dτb
            C[end] = 0

            for j = 1:N
                fill!(RHS, 0.0)
                RHS[j] = dS_dT[j]

                if j == N
                    RHS[end] += (2μ/3) * grad_S
                end

                solve_feautrier!(δu, A, B, C, RHS, dl, d, du)

                @inbounds for i = 2:N-1
                    δH = μ * (δu[i+1] - δu[i-1]) / (Δτ[i] + Δτ[i-1])
                    δF = 4π * δH
                    Jmat[i,j] += bw * wμ * δF
                end
            end
        end
    end
end

function add_convection_to_jacobian!(
    Jmat,
    dFconv_dT,
    dFconv_dT_minus,
    conv_start
)
    N = size(Jmat,1)
    for i = conv_start:N
        Jmat[i,i]     += 0.25 * dFconv_dT[i]
        Jmat[i,i-1]   += 0.25 * dFconv_dT_minus[i]
    end
end

function update_temperature_fully_linearized!(
    T, dT,
    F_rad, F_conv,
    Jmat,
    Teff
)
    F_target = σ_SB * Teff^4
    R = F_rad .+ F_conv .- F_target

    dT .= - (Jmat \ R)

    for i in eachindex(T)
        if dT[i] * T[i] < 0
            dT[i] *= 0.75
        end
        dTmax = 0.08 * T[i]
        dT[i] = clamp(dT[i], -dTmax, dTmax)
        T[i] += dT[i]
    end
end





function compute_diagonal_inv2!(diag_inv, A, B, C)
    n = length(B)
    
    # 1. Forward Sweep
    # We use 'diag_inv' to store the D terms temporarily to avoid allocation
    diag_inv[1] = B[1]
    
    @inbounds for k in 2:n
        # D[k] = B[k] - A[k] * C[k-1] / D[k-1]
        val = A[k] * C[k-1] / diag_inv[k-1]
        diag_inv[k] = B[k] - val
    end
    
    # 2. Backward Sweep
    # Overwrite 'diag_inv' with the actual inverse diagonal elements
    # Z[n] = 1 / D[n]
    diag_inv[n] = 1.0 / diag_inv[n]
    
    @inbounds for k in (n-1):-1:1
        Z_next = diag_inv[k+1]     # Already computed inverse of next element
        D_curr = diag_inv[k]       # Current D term (from forward sweep)
        
        # Z[k] = (1/D[k]) * (1 + A[k+1] * C[k] * Z[k+1])
        diag_inv[k] = (1.0 / D_curr) * (1.0 + A[k+1] * C[k] * Z_next)
    end
end




function compute_temperature_corrections_auer_mihalas!(dT, T, ρ, z, 
                                         eos, opa, T_eff;
                                         μ_nodes, μ_weights, J_comp, 
                                         λ_weights=nothing)
    
    N_d = length(T)
    N_f = length(opa.opa.λ) 
    N_a = length(μ_nodes)

    scale = 0.5 / sum(μ_weights)
    μ_weights_scaled = μ_weights .* scale
    
    opacity_ratio = zeros(N_d, N_f)
    B = zeros(N_d, N_f)
    dB_dT = zeros(N_d, N_f)
    χ_std = zeros(N_d)
    
    w_ν = isnothing(λ_weights) ? ones(N_f) ./ N_f : λ_weights
    
    lnρ = log.(ρ)
    lnT = log.(T)
    
    for d in 1:N_d
        χ_std[d] = exp(lookup(eos.eos, :lnRoss, lnρ[d], lnT[d])) * ρ[d]
        
        for f in 1:N_f
            χ_ν = lookup(eos.eos, opa.opa, :κ, lnρ[d], lnT[d], f) 
            B[d, f] = lookup(eos.eos, opa.opa, :src, lnρ[d], lnT[d], f)
            dB_dT[d, f] = TSO.extended_lookup(eos.eos, opa, :dS_dT, lnρ[d], lnT[d], f)
            
            if χ_std[d] > 0
                opacity_ratio[d, f] = χ_ν / χ_std[d]
            else
                opacity_ratio[d, f] = χ_ν 
            end
        end
    end
    
    τ = zeros(N_d)
    τ[1] = 0.0 
    for d in 2:N_d
        dz = abs(z[d] - z[d-1])
        avg_χ = 0.5 * (χ_std[d] + χ_std[d-1])
        τ[d] = τ[d-1] + avg_χ * dz
    end
    J = solve_rt_auer_mihalas(τ, μ_nodes, μ_weights_scaled, w_ν, opacity_ratio, B, dB_dT, T_eff)
    J_mean = similar(J_comp)
    J_mean .= 0.0
    #dT = zeros(N_d)
    
    for d in 1:N_d
        numerator = 0.0
        denominator = 0.0
        valBsum = 0.0
        valdBsum = 0.0
        Jsum = 0.0
        
        for f in 1:N_f
            η = opacity_ratio[d, f]
            val_B = B[d, f]
            val_dB = dB_dT[d, f]
            w_freq = w_ν[f]
            
            J_mean_angle = 0.0
            for a in 1:N_a
                J_mean_angle += μ_weights_scaled[a] * J[d, a, f]
            end

            J_mean[d] += w_freq * J_mean_angle
            
            weight_factor = w_freq * η
            numerator += weight_factor * (J_mean_angle - val_B)
            denominator += weight_factor * val_dB
            valBsum += weight_factor * val_B
            valdBsum += weight_factor * val_dB
            Jsum += weight_factor * J_mean_angle
        end
        
        if (d == 1) || (d == N_d)
            @show d,numerator, denominator, valBsum, valdBsum, Jsum
        end
        if abs(denominator) > 1e-30
            dT[d] = numerator / denominator
        else
            dT[d] = 0.0
        end
        
        max_correction = 0.1 * T[d] 
        dT[d] = clamp(dT[d], -max_correction, max_correction)
    end

    @show minimum(dT), maximum(dT)
    @show minimum(J_comp), maximum(J_comp)
    @show minimum(J_mean), maximum(J_mean)
    
    return dT
end

function solve_rt_auer_mihalas(τ, μ, w_μ, w_ν, opacity, B, dB_dT, T_eff)
    
    # Ensure this constant matches your B units (CGS)
    σ_SB = 5.670374419e-5 
    H_flux = (σ_SB * T_eff^4) / (4.0 * π)

    N_d, N_f = size(opacity)
    N_a = length(μ)
    N_state = N_a * N_f 

    # Flattened Arrays
    ω = vec([w_μ[a] * w_ν[f] for a in 1:N_a, f in 1:N_f])
    μ_flat = repeat(μ, outer=N_f)
    
    # Use simple permutedims/reshape to ensure correct layout
    η_flat = reshape(permutedims(repeat(opacity, inner=(1, 1, N_a)), (1, 3, 2)), N_d, N_state)
    B_flat = reshape(permutedims(repeat(B, inner=(1, 1, N_a)), (1, 3, 2)), N_d, N_state)
    dB_flat = reshape(permutedims(repeat(dB_dT, inner=(1, 1, N_a)), (1, 3, 2)), N_d, N_state)

    Blocks_A = [zeros(N_state, N_state) for _ in 1:N_d]
    Blocks_B = [zeros(N_state, N_state) for _ in 1:N_d]
    Blocks_C = [zeros(N_state, N_state) for _ in 1:N_d]
    Vectors_R = [zeros(N_state) for _ in 1:N_d]

    for d in 1:N_d
        η_d = η_flat[d, :]
        B_d = B_flat[d, :]
        dB_d = dB_flat[d, :]
        
        Sum_denom = dot(ω, η_d .* dB_d); Sum_denom = (Sum_denom == 0) ? 1.0 : Sum_denom
        v_c = dB_d ./ Sum_denom
        Sum_wnB = dot(ω, η_d .* B_d)

        # Standard geometric Δτ
        Δτ_minus = (d > 1)   ? (τ[d] - τ[d-1]) : 0.0
        Δτ_plus  = (d < N_d) ? (τ[d+1] - τ[d]) : 0.0
        
        for i in 1:N_state
            # --- FIX: SCALE Δτ BY OPACITY (Δτ_ν = η * Δτ) ---
            # We use the average η between the two depth points
            η_val = η_d[i]
            
            if d > 1
                Δτ_nu_minus = Δτ_minus * 0.5 * (η_val + η_flat[d-1, i])
            else
                Δτ_nu_minus = 0.0
            end
            
            if d < N_d
                Δτ_nu_plus = Δτ_plus * 0.5 * (η_val + η_flat[d+1, i])
            else
                Δτ_nu_plus = 0.0
            end

            # Average for 2nd derivative denominator
            if d == 1
                Δτ_nu_avg = Δτ_nu_plus
            elseif d == N_d
                Δτ_nu_avg = Δτ_nu_minus
            else
                Δτ_nu_avg = 0.5 * (Δτ_nu_minus + Δτ_nu_plus)
            end
            
            # Use SCALED steps for coefficients
            val_u = (μ_flat[i]^2) / Δτ_nu_avg
            u_minus = (d > 1)   ? val_u / Δτ_nu_minus : 0.0
            u_plus  = (d < N_d) ? val_u / Δτ_nu_plus  : 0.0
            
            if d > 1;     Blocks_A[d][i, i] = u_minus; end
            if d < N_d;   Blocks_C[d][i, i] = u_plus;  end
            
            Blocks_B[d][i, i] = (u_minus + u_plus) + 1.0
            Vectors_R[d][i] = B_d[i] - v_c[i] * Sum_wnB
        end
        
        Blocks_B[d] .-= v_c * (ω .* η_d)'
    end

    # Surface BC (τ=0)
    d = 1
    Δτ_1 = τ[2] - τ[1]
    for i in 1:N_state
        # Scale Δτ_1 by η
        Δτ_nu_1 = Δτ_1 * 0.5 * (η_flat[1, i] + η_flat[2, i])
        
        Blocks_A[d][i, :] .= 0; Blocks_B[d][i, :] .= 0; Blocks_C[d][i, :] .= 0; Vectors_R[d][i] = 0
        
        term = μ_flat[i] / Δτ_nu_1
        Blocks_B[d][i, i] = 1.0 + term
        Blocks_C[d][i, i] = term
    end

    # Lower BC (τ=τ_max)
    d = N_d
    Δτ_N = τ[N_d] - τ[N_d-1]
    η_N = η_flat[N_d, :]
    B_N = B_flat[N_d, :]; dB_N = dB_flat[N_d, :]
    
    Sum_wn_dB = dot(ω, η_N .* dB_N)
    Sum_wmu_eta_dB = dot(ω, (μ_flat ./ η_N) .* dB_N)
    Sum_wmu_B = dot(ω, μ_flat .* B_N)
    Sum_wmu_dB = dot(ω, μ_flat .* dB_N)
    Sum_wn_B = dot(ω, η_N .* B_N)

    H_term = H_flux - Sum_wmu_B + (Sum_wmu_dB * Sum_wn_B / Sum_wn_dB)

    for i in 1:N_state
        Blocks_A[d][i, :] .= 0; Blocks_B[d][i, :] .= 0; Blocks_C[d][i, :] .= 0; Vectors_R[d][i] = 0
        
        # Scale Δτ_N by η
        Δτ_nu_N = Δτ_N * 0.5 * (η_N[i] + η_flat[d-1, i])
        
        mu_dt = μ_flat[i] / Δτ_nu_N
        Blocks_A[d][i, i] = mu_dt 
        Blocks_B[d][i, i] = mu_dt + 1.0 
        
        factor1 = dB_N[i] / Sum_wn_dB
        Blocks_B[d][i, :] .-= factor1 .* (ω .* η_N)
        Vectors_R[d][i] += B_N[i] - factor1 * Sum_wn_B
        
        C_flux = (μ_flat[i] / η_N[i]) * dB_N[i] / Sum_wmu_eta_dB
        
        Vectors_R[d][i] += C_flux * H_term
        Blocks_B[d][i, :] .-= C_flux .* (ω .* μ_flat)
        Blocks_B[d][i, :] .+= (C_flux * Sum_wmu_dB / Sum_wn_dB) .* (ω .* η_N)
    end

    Es = Vector{Matrix{Float64}}(undef, N_d)
    Fs = Vector{Vector{Float64}}(undef, N_d)
    
    Es[1] = Blocks_B[1] \ Blocks_C[1]
    Fs[1] = Blocks_B[1] \ Vectors_R[1]
    
    for d in 2:N_d
        M_temp = Blocks_B[d] - Blocks_A[d] * Es[d-1]
        if d < N_d; Es[d] = M_temp \ Blocks_C[d]; end
        Fs[d] = M_temp \ (Vectors_R[d] + Blocks_A[d] * Fs[d-1])
    end
    
    J_flat = Vector{Vector{Float64}}(undef, N_d)
    J_flat[N_d] = Fs[N_d]
    for d in (N_d-1):-1:1
        J_flat[d] = Es[d] * J_flat[d+1] + Fs[d]
    end

    # Reshape back to (Depth, Angle, Freq)
    J_out = zeros(N_d, N_a, N_f)
    idx(ia, ifreq) = (ifreq - 1) * N_a + ia
    for d in 1:N_d
        for ifreq in 1:N_f
            for ia in 1:N_a
                J_out[d, ia, ifreq] = J_flat[d][idx(ia, ifreq)]
            end
        end
    end

    return J_out
end

