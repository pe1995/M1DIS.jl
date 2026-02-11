
#=function generate_mu_grid(n_points::Integer)
    μ_grid, μ_weights = gausslegendre(n_points)
    μ_grid = @. μ_grid / 2 + 0.5
    μ_weights ./= 2
    μ_grid, μ_weights
end=#

function generate_mu_grid(n_points::Integer)
    x, w = gausslegendre(n_points)
    return @. x / 2 + 0.5, @. w / 2
end








function compute_diagonal_inv!(diag_inv, A, B, C)
    n = length(B)
    
    D = deepcopy(diag_inv)
    D[1] = B[1]
    @inbounds for k in 2:n
        val = A[k] * C[k-1] / D[k-1] 
        D[k] = B[k] - val
    end
    D[n] = 1.0 / D[n]
    
    @inbounds for k in (n-1):-1:1
        Z_next = D[k+1]
        D_curr = D[k]
        D[k] = (1.0 / D_curr) * (1.0 + A[k+1] * C[k] * Z_next)
    end
end

@inline function trace_ray(range_iter, I_start, τ_vert, S_cell, abs_μ)
    I_curr = I_start
    
    # Iterate through the cells along the path
    @inbounds for k in range_iter
        # Optical depth difference of this cell
        # (Works for both up and down if indices are managed correctly by caller)
        Δτ_vertical = abs(τ_vert[k+1] - τ_vert[k])
        Δτ = Δτ_vertical / abs_μ

        # Second-order expansion for small optical depths for stability
        trans = if Δτ < 1e-4
            1.0 - Δτ + 0.5 * Δτ^2
        else
            exp(-Δτ)
        end
        
        S_c = S_cell[k]
        I_curr = I_curr * trans + S_c * (1.0 - trans)
    end
    return I_curr
end

function update_radiation_z_longchar!(J, F, g_rad; T, ρ, z, eos, opa, μ_weights=nothing, μ_angles=nothing, λ_weights=nothing, irradiation=nothing) 
    Nnodes = length(z)
    ncells = Nnodes - 1
    Δz = diff(z)                 

    use_angles, use_weights = if isnothing(μ_weights) || isnothing(μ_angles)
        generate_mu_grid(4)
    else
        copy(μ_angles), copy(μ_weights)
    end
    use_weights .*= (0.5 / sum(use_weights))

    lnrho = log.(ρ)
    lnt = log.(T)
    
    J_nu = zeros(eltype(T), size(T))
    H_nu = zeros(eltype(T), size(T))
    S_nodes = similar(T)
    k_rho_nodes = similar(T)
    S_cell = zeros(eltype(T), ncells)
    k_cell = zeros(eltype(T), ncells)
    τ_vert = zeros(eltype(T), Nnodes)

    # Reset Global Outputs
    J .= 0.0; F .= 0.0; g_rad .= 0.0

    nbin = length(opa.λ)
    bin_weights = isnothing(λ_weights) ? ones(nbin) : λ_weights

    for (bin, bw) in enumerate(bin_weights)
        Irr = isnothing(irradiation) ? 0.0 : irradiation[bin]
        S_nodes .= lookup(eos, opa, :src, lnrho, lnt, bin)
        k_rho_nodes .= lookup(eos, opa, :κ, lnrho, lnt, bin)

        compute_τ_grid!(τ_vert; z=z, ρκ=k_rho_nodes)
        @inbounds for i in 1:ncells
            S_cell[i] = 0.5 * (S_nodes[i] + S_nodes[i+1])
            k_cell[i] = 0.5 * (k_rho_nodes[i] + k_rho_nodes[i+1])
        end

        if Nnodes > 1
             dS = S_nodes[end] - S_nodes[end-1]
             dz = z[end] - z[end-1] # Negative
             dtau_dz = k_rho_nodes[end]
             
             grad_S = -(dS / dz) / dtau_dz
             
             if grad_S < 0
                 grad_S = 0.0
             end
        else
            dS_bot = S_nodes[end] - S_nodes[end-1]
            dt_bot = k_cell[end] * Δz[end]
            grad_S = dt_bot > 1e-30 ? (dS_bot / dt_bot) : 0.0
        end

        J_nu .= 0.0
        H_nu .= 0.0
        
        for (μ, wμ) in zip(use_angles, use_weights)
            abs_μ = abs(μ)

            # solve for the intensity at every node 'target_i' independently
            for target_i in 1:Nnodes
                # 1. Downward Ray (Top -> target_i)
                # Range: Cells 1 to target_i-1
                I_down = if target_i == 1
                    trans_top = exp(-τ_vert[1] / abs_μ)
                    S_nodes[1] * (1.0 - trans_top) + Irr
                else
                    trace_ray(1:(target_i-1), 0.0, τ_vert, S_cell, abs_μ)
                end

                # 2. Upward Ray (Bottom -> target_i)
                # Range: Cells N-1 down to target_i
                I_bottom_bc = S_nodes[end] + (abs_μ * grad_S)
                I_up = if target_i == Nnodes
                    I_bottom_bc
                else
                    trace_ray(ncells:-1:target_i, I_bottom_bc, τ_vert, S_cell, abs_μ)
                end
                
                J_nu[target_i] += wμ * (I_up + I_down)
                H_nu[target_i] += wμ * μ * (I_up - I_down)
            end
        end

        @inbounds for i in eachindex(J)
            F_bin = bw * (4π * H_nu[i])
            J[i] += bw * J_nu[i]
            F[i] += F_bin
            g_rad[i] += k_rho_nodes[i] / ρ[i] * F_bin / c_light
        end
    end
    
    # Enforce Monotonicity of Radiative Flux (User Requested)
    # F_rad must decrease with depth (index increase) as convection takes over.
    #=for i in 2:Nnodes
        if F[i] > F[i-1]
            F[i] = F[i-1]
        end
    end=#
end







#= Parallel version ---> Needs update! =#

function _radiation_chunk_kernel(bin_range, T, ρ, z, eos, opa, 
                                μ_angles, μ_weights_scaled, bin_weights, 
                                lnrho, lnt, Δz, ncells, irradiation)
    J_nu = zeros(Float64, size(T))
    H_nu = zeros(Float64, size(T))
    Q_chunk    = zeros(Float64, size(T))
    dQdT_chunk = zeros(Float64, size(T))
    
    S_nodes = zeros(Float64, size(J_nu))
    dBdT_nodes = zeros(Float64, size(J_nu))
    k_rho_nodes = zeros(Float64, size(J_nu))
    S_cell = zeros(Float64, ncells)
    k_cell = zeros(Float64, ncells)
    
    # Pre-allocate τ_vert and trace arrays if needed, but they are small (Nnodes)
    Nnodes = length(T)
    τ_vert = zeros(Float64, Nnodes)

    J_chunk = zeros(Float64, size(T))
    F_chunk = zeros(Float64, size(T))
    g_chunk = zeros(Float64, size(T))

    @inbounds for bin in bin_range
        bw = bin_weights[bin]
        Irr = isnothing(irradiation) ? 0.0 : irradiation[bin]

        S_nodes .= lookup(eos, opa.opa, :src, lnrho, lnt, bin)
        dBdT_nodes .= TSO.extended_lookup(eos, opa, :dS_dT, lnrho, lnt, bin)
        k_rho_nodes .= lookup(eos, opa.opa, :κ, lnrho, lnt, bin)
        
        # Compute τ_vert for this bin (needed for trace_ray)
        # We can reuse the serial logic: compute_τ_grid!
        # But we need to define it or inline it. It is defined in _RT.jl? 
        # Yes, compute_τ_grid! is likely available in the module scope.
        compute_τ_grid!(τ_vert; z=z, ρκ=k_rho_nodes)

        @inbounds for i in 1:ncells
            S_cell[i] = 0.5 * (S_nodes[i] + S_nodes[i+1])
            k_cell[i] = 0.5 * (k_rho_nodes[i] + k_rho_nodes[i+1])
        end

        # --- Bottom Boundary Condition (Geometric Gradient) ---
        if Nnodes > 1
             dS = S_nodes[end] - S_nodes[end-1]
             dz = z[end] - z[end-1] # Negative
             dtau_dz = k_rho_nodes[end]
             
             # grad_S = (dS/dz) / (-κρ)
             # grad_S = (dS/dz) / (-dtau_dz)
             grad_S = -(dS / dz) / dtau_dz
             grad_S = dS
             if grad_S < 0
                 grad_S = 0.0
             end
        else
            dS_bot = S_nodes[end] - S_nodes[end-1]
            dt_bot = k_cell[end] * Δz[end]
            #grad_S = dt_bot > 1e-30 ? (dS_bot / dt_bot) : 0.0
            grad_S = dt_bot > 1e-30 ? dS_bot : 0.0
        end

        J_nu .= 0.0
        H_nu .= 0.0

        # Angular integration
        for (μ, wμ) in zip(μ_angles, μ_weights_scaled)
            abs_μ = abs(μ)
            
            # solve for the intensity at every node 'target_i' independently
            # (Matches serial logic)
            for target_i in 1:Nnodes
                # 1. Downward Ray (Top -> target_i)
                I_down = if target_i == 1
                    trans_top = exp(-(k_cell[1] * Δz[1]) / abs_μ)
                    S_nodes[1] * (1.0 - trans_top) + Irr
                else
                    trace_ray(1:(target_i-1), 0.0, τ_vert, S_cell, abs_μ)
                end

                # 2. Upward Ray (Bottom -> target_i)
                I_bottom_bc = S_nodes[end] + (abs_μ * grad_S)
                I_up = if target_i == Nnodes
                    I_bottom_bc
                else
                    trace_ray(ncells:-1:target_i, I_bottom_bc, τ_vert, S_cell, abs_μ)
                end
                
                J_nu[target_i] += wμ * (I_up + I_down)
                H_nu[target_i] += wμ * μ * (I_up - I_down)
            end
        end

        @inbounds for i in eachindex(J_chunk)
            F_bin = bw * (4π * H_nu[i])
            J_chunk[i] += bw * J_nu[i]
            F_chunk[i] += F_bin
            g_chunk[i] += k_rho_nodes[i] / ρ[i] * F_bin / c_light

            Q_chunk[i]    += bw * k_rho_nodes[i] * (J_nu[i] - S_nodes[i])
            dQdT_chunk[i] += bw * k_rho_nodes[i] * dBdT_nodes[i]
        end
    end

    return (J_chunk, F_chunk, g_chunk, Q_chunk, dQdT_chunk)
end

function update_radiation_z_longchar_dagger!(J, F, g_rad, Q, dQdT; T, ρ, z, eos, opa,
                                  μ_weights=nothing,
                                  μ_angles=nothing,
                                  λ_weights=nothing, irradiation=nothing) 
    
    Nnodes = length(z)
    ncells = Nnodes - 1
    Δz = diff(z)
    
    lnrho = log.(ρ)
    lnt = log.(T)

    use_angles, use_weights = if isnothing(μ_weights) || isnothing(μ_angles)
        generate_mu_grid(4)
    else
        copy(μ_angles), copy(μ_weights)
    end
    
    scale = 0.5 / sum(use_weights)
    μ_weights_scaled = use_weights .* scale

    nbin = length(opa.opa.λ)
    bin_weights = isnothing(λ_weights) ? ones(nbin) : λ_weights

    n_workers = Base.Threads.nthreads() 
    chunk_size = max(1, cld(nbin, n_workers))
    chunks = Iterators.partition(1:nbin, chunk_size)

    tasks = map(chunks) do range
        Dagger.@spawn _radiation_chunk_kernel(
            range, T, ρ, z, eos, opa, 
            use_angles, μ_weights_scaled, bin_weights, 
            lnrho, lnt, Δz, ncells, irradiation
        )
    end

    fill!(J, 0.0)
    fill!(F, 0.0)
    fill!(g_rad, 0.0)
    fill!(Q, 0.0)
    fill!(dQdT, 0.0)
    results = fetch.(tasks) 

    for (J_part, F_part, g_part, Q_part, dQdT_part) in results
        J .+= J_part
        F .+= F_part
        g_rad .+= g_part
        Q .+= Q_part
        dQdT .+= dQdT_part
    end
    
    # Enforce Monotonicity of Radiative Flux 
    #=for i in 2:Nnodes
        if F[i] > F[i-1]
            F[i] = F[i-1]
            g_rad[i] = g_rad[i-1]
            J[i] = J[i-1]
        end
    end=#

    return nothing
end




function compute_opacities(eos, opa, T, ρ)
    chi = zeros(Float64, length(opa.opa.λ), length(T))
    chi_ref = zeros(Float64, length(T))
    B = zeros(Float64, length(opa.opa.λ), length(T))
    dBdT = zeros(Float64, length(opa.opa.λ), length(T))

    compute_opacities!(chi, chi_ref, B, dBdT, eos, opa, T, ρ)

    return chi, chi_ref, B, dBdT
end

function compute_opacities!(chi, chi_ref, B, dBdT, eos, opa, T, ρ)
    lnrho = log.(ρ)
    lnt = log.(T)
    
    Threads.@threads for i in eachindex(opa.opa.λ)
        chi[i, :] .= lookup(eos.eos, opa.opa, :κ, lnrho, lnt, i)
        B[i, :] .= lookup(eos.eos, opa.opa, :src, lnrho, lnt, i)
        dBdT[i, :] .= TSO.extended_lookup(eos.eos, opa, :dS_dT, lnrho, lnt, i)
    end
    chi_ref .= exp.(lookup(eos.eos, :lnRoss, lnrho, lnt)) .* ρ

    return nothing
end

# ==============================================================================
# Bilinear Interpolation for large frequency grids
# ==============================================================================
struct InterpCoefs
    idx::Int        
    w_low::Float64  
    w_high::Float64 
end

function get_interp_coefs(grid_nodes, val)
    if val <= first(grid_nodes)
        return InterpCoefs(1, 1.0, 0.0)
    elseif val >= last(grid_nodes)
        return InterpCoefs(length(grid_nodes)-1, 0.0, 1.0)
    end
    
    i = searchsortedlast(grid_nodes, val)
    
    x0 = grid_nodes[i]
    x1 = grid_nodes[i+1]
    w_high = (val - x0) / (x1 - x0)
    w_low  = 1.0 - w_high
    
    return InterpCoefs(i, w_low, w_high)
end

function _compute_opacities_chunk!(range, chi, B, dBdT, opa, coefs_T, coefs_Rho)
    kappa_data = opa.opa.κ
    src_data   = opa.opa.src
    dSdT_data  = opa.extensions[:dS_dT]

    n_shells = length(coefs_T)

    @inbounds for i in range
        for j in 1:n_shells
            ct = coefs_T[j]
            cr = coefs_Rho[j]
            
            it, ir = ct.idx, cr.idx
            
            w00 = ct.w_low  * cr.w_low
            w10 = ct.w_high * cr.w_low
            w01 = ct.w_low  * cr.w_high
            w11 = ct.w_high * cr.w_high
            
            val_k = w00 * log(kappa_data[it,   ir,   i]) +
                    w10 * log(kappa_data[it+1, ir,   i]) +
                    w01 * log(kappa_data[it,   ir+1, i]) +
                    w11 * log(kappa_data[it+1, ir+1, i])
            chi[i, j] = exp(val_k)

            val_s = w00 * log(src_data[it,   ir,   i]) +
                    w10 * log(src_data[it+1, ir,   i]) +
                    w01 * log(src_data[it,   ir+1, i]) +
                    w11 * log(src_data[it+1, ir+1, i])
            B[i, j] = exp(val_s)

            val_d = w00 * dSdT_data[it,   ir,   i] +
                    w10 * dSdT_data[it+1, ir,   i] +
                    w01 * dSdT_data[it,   ir+1, i] +
                    w11 * dSdT_data[it+1, ir+1, i]
            dBdT[i, j] = val_d
        end
    end
end

function compute_opacities_chunked!(chi, chi_ref, B, dBdT, eos, opa, T, ρ)
    lnrho = log.(ρ)
    lnt = log.(T)
    
    grid_T = eos.eos.lnT
    grid_Rho = eos.eos.lnRho
    n_shells = length(lnrho)
    
    coefs_T   = Vector{InterpCoefs}(undef, n_shells)
    coefs_Rho = Vector{InterpCoefs}(undef, n_shells)
    
    for j in 1:n_shells
        coefs_T[j]   = get_interp_coefs(grid_T, lnt[j])
        coefs_Rho[j] = get_interp_coefs(grid_Rho, lnrho[j])
    end

    λ_indices = eachindex(opa.opa.λ)
    chunk_size = cld(length(λ_indices), Threads.nthreads())
    
    tasks = map(Iterators.partition(λ_indices, chunk_size)) do range
        Threads.@spawn _compute_opacities_chunk!(range, chi, B, dBdT, opa, coefs_T, coefs_Rho)
    end
    
    chi_ref .= exp.(lookup(eos.eos, :lnRoss, lnrho, lnt)) .* ρ
    
    wait.(tasks)
    return nothing
end
