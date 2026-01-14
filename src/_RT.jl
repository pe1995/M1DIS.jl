
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

"""
    compute_diagonal_inv!(diag_inv, A, B, C)

Compute the diagonal elements of the inverse of a Tridiagonal matrix `T` efficienty in O(N).
`T` is defined by lower diagonal `A`, diagonal `B`, and upper diagonal `C`.
Note: `A` is indexed such that `A[i] = T[i, i-1]`.
"""
function compute_diagonal_inv!(diag_inv, A, B, C)
    n = length(B)
    
    # Forward sweep for Pivots D_k
    # D_1 = B_1
    # D_k = B_k - A_k * C_{k-1} / D_{k-1}
    
    # We use diag_inv to store D temporarily
    D = diag_inv 
    
    D[1] = B[1]
    @inbounds for k in 2:n
        val = A[k] * C[k-1] / D[k-1] 
        D[k] = B[k] - val
    end
    
    # Backward sweep for Diagonal Elements Z_k
    # Z_N = 1/D_N
    # Z_k = (1/D_k) * (1 + A_{k+1} * C_k * Z_{k+1})
    
    # Last element
    D[n] = 1.0 / D[n]
    
    @inbounds for k in (n-1):-1:1
        Z_next = D[k+1]
        D_curr = D[k]
        
        # Careful: D[k] is being overwritten, so we needed D_curr
        D[k] = (1.0 / D_curr) * (1.0 + A[k+1] * C[k] * Z_next)
    end
end

"""
    trace_ray(range_iter, I_start, τ_vert, S_cell, abs_μ)

Integrates the radiative transfer equation along a ray path defined by `range_iter`.
"""
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

function update_radiation_z_longchar!(J, F, g_rad; T, ρ, z, eos, opa,
                                      μ_weights=nothing,
                                      μ_angles=nothing,
                                      λ_weights=nothing, irradiation=nothing) 
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
             # 1st Order Geometric Gradient (Robust)
             # dS/dτ = (dS/dz) / (-κρ)
             
             dS = S_nodes[end] - S_nodes[end-1]
             dz = z[end] - z[end-1] # Negative
             dtau_dz = k_rho_nodes[end]
             
             # grad_S = (dS/dz) / (-κρ)
             # grad_S = (dS/dz) / (-dtau_dz)
             grad_S = -(dS / dz) / dtau_dz
             
             if grad_S < 0
                 grad_S = 0.0
             end
        else
            dS_bot = S_nodes[end] - S_nodes[end-1]
            dt_bot = k_cell[end] * Δz[end]
            grad_S = dt_bot > 1e-30 ? (dS_bot / dt_bot) : 0.0
        end

        #grad_S = S_nodes[end] - S_nodes[end-1]

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
    for i in 2:Nnodes
        if F[i] > F[i-1]
            F[i] = F[i-1]
        end
    end
end


"""
    update_radiation_z_feutrier!(J, F, g_rad; T, ρ, z, eos, opa,
                                 μ_weights=nothing, μ_angles=nothing,
                                 λ_weights=nothing, irradiation=nothing,
                                 diagonal_inv_operator=nothing,
                                 flux_jacobian=nothing) 

Solve the radiative transfer equation using the Feutrier method (2nd order).
If `diagonal_inv_operator` is provided (vector), it fills the Lambda-diagonal.
"""
function update_radiation_z_feutrier!(J, F, g_rad; T, ρ, z, eos, opa,
                                      μ_weights=nothing,
                                      μ_angles=nothing,
                                      λ_weights=nothing, irradiation=nothing,
                                      diagonal_inv_operator=nothing) 
    Nnodes = length(z)
    ncells = Nnodes - 1

    use_angles, use_weights = generate_mu_grid(3)
    
    # Normalize weights to sum to 1.0 (Essential for J calculation)
    w_sum = sum(use_weights)
    if abs(w_sum - 1.0) > 1e-6
         use_weights .*= (1.0 / w_sum)
    end

    lnrho = log.(ρ)
    lnt = log.(T)
    
    J_nu = zeros(eltype(T), size(T))
    H_nu = zeros(eltype(T), size(T))
    S_nodes = similar(T)
    k_rho_nodes = similar(T)
    
    # ... (rest of RT logic) ...

    # Removed misplaced flux calculation block.
    k_rho_nodes = similar(T)
    τ_vert = zeros(eltype(T), Nnodes)
    Δτ = zeros(eltype(T), Nnodes-1)

    # Temporary arrays for the Tridiagonal solver
    A = zeros(eltype(T), Nnodes) 
    B = zeros(eltype(T), Nnodes) 
    C = zeros(eltype(T), Nnodes) 
    RHS = zeros(eltype(T), Nnodes)
    u_solution = zeros(eltype(T), Nnodes)
    v_solution = zeros(eltype(T), Nnodes) 
    dl = zeros(eltype(T), Nnodes-1) 
    du = zeros(eltype(T), Nnodes-1) 
    d  = zeros(eltype(T), Nnodes) 
    
    diag_inv_buffer = zeros(eltype(T), Nnodes)
    
    J .= 0.0; F .= 0.0; g_rad .= 0.0
    
    if !isnothing(diagonal_inv_operator)
        fill!(diagonal_inv_operator, 0.0)
    end

    nbin = length(opa.λ)
    bin_weights = isnothing(λ_weights) ? ones(nbin) : λ_weights

    for (bin, bw) in enumerate(bin_weights)
        Irr = isnothing(irradiation) ? 0.0 : irradiation[bin]
        S_nodes .= lookup(eos, opa, :src, lnrho, lnt, bin)
        k_rho_nodes .= lookup(eos, opa, :κ, lnrho, lnt, bin)
        
        compute_τ_grid!(τ_vert; z=z, ρκ=k_rho_nodes)
        Δτ .= diff(τ_vert) 
        
        if Nnodes > 2
            # 2nd Order Backward Difference for Bottom Boundary Gradient
            # Fit parabola S(τ) to last 3 points
            S3, S2, S1 = S_nodes[end], S_nodes[end-1], S_nodes[end-2]
            dτ2 = Δτ[end]   # τ_N - τ_{N-1}
            dτ1 = Δτ[end-1] # τ_{N-1} - τ_{N-2}
            
            # Derivative at N (S3)
            # Formula for non-uniform grid:
            # f'(x3) = (f2 - f3) * (x2 - x3) / ( (x1 - x2)(x1 - x3) ) ... No, let's use the explicit form
            
            # Weighted slope:
            # P(τ) = a(τ - τN)^2 + b(τ - τN) + c
            # c = SN
            # S_{N-1} = a(-dτ2)^2 + b(-dτ2) + c
            # S_{N-2} = a(-(dτ1+dτ2))^2 + b(-(dτ1+dτ2)) + c
            
            # Linear system for a, b. We need b = S'(τN).
            
            # Let h1 = dτ2 (dist N to N-1)
            # Let h2 = dτ1 + dτ2 (dist N to N-2)
            
            h1 = dτ2
            h2 = dτ1 + dτ2
            
            # S_{N-1} - S_N = a*h1^2 - b*h1  (using positive h distances backwards) -> Wait, let's be careful with signs.
            # Local coordinate x = τ - τ_N. 
            # node N: x=0, S=S3
            # node N-1: x=-h1, S=S2
            # node N-2: x=-h2, S=S1
            
            # S2 = a*h1^2 - b*h1 + S3
            # S1 = a*h2^2 - b*h2 + S3
            
            # (S2 - S3) = a*h1^2 - b*h1  => eq1
            # (S1 - S3) = a*h2^2 - b*h2  => eq2
            
            # Multiply eq1 by h2^2, eq2 by h1^2:
            # h2^2(S2-S3) = a*h1^2*h2^2 - b*h1*h2^2
            # h1^2(S1-S3) = a*h1^2*h2^2 - b*h2*h1^2
            
            # Subtract:
            # h2^2(S2-S3) - h1^2(S1-S3) = -b * (h1*h2^2 - h2*h1^2) = -b * h1*h2*(h2 - h1)
            
            # b = - [ h2^2(S2-S3) - h1^2(S1-S3) ] / [ h1*h2*(h2 - h1) ]
            # b = [ h2^2(S3-S2) - h1^2(S3-S1) ] / [ h1*h2*(h2 - h1) ]
            
            term1 = h2^2 * (S3 - S2)
            term2 = h1^2 * (S3 - S1)
            denom = h1 * h2 * (h2 - h1)
            
            grad_S = (term1 - term2) / denom
        else
            # Fallback for tiny grids
            dS_bot = S_nodes[end] - S_nodes[end-1]
            dt_bot = Δτ[end]
            grad_S = dt_bot > 1e-30 ? (dS_bot / dt_bot) : 0.0
        end
        
        J_nu .= 0.0
        H_nu .= 0.0

        for (μ, wμ) in zip(use_angles, use_weights)
            abs_μ = abs(μ)
            μ2 = abs_μ^2
            
            # --- Build the Tridiagonal System ---
            @inbounds for k in 2:(Nnodes-1)
                dτ_minus = Δτ[k-1]
                dτ_plus  = Δτ[k]
                
                denom = dτ_minus + dτ_plus
                factor = 2.0 * μ2 / denom
                
                A[k] = factor / dτ_minus      
                C[k] = factor / dτ_plus       
                B[k] = 1.0 + factor * (1.0/dτ_plus + 1.0/dτ_minus)
                RHS[k] = S_nodes[k]
            end
            
            # --- Boundary Conditions ---
            # 2nd Order Auer (1967) Condition with Inward Intensity
            inv_dtau = 1.0 / Δτ[1]
            two_mu_dtau = 2.0 * abs_μ * inv_dtau
            
            B[1] = 1.0 + two_mu_dtau
            C[1] = two_mu_dtau - 1.0
            
            # Estimate Inward Intensity from layers above the grid (Soft Boundary)
            # REVERTED: Calculating I_inc from extrapolation leads to excessive heating (~10,000K surface)
            # with the current Unsold-Mawe solver. We revert to Vacuum BC (I_inc = 0 + Irr) for stability.
            # extrap_tau = τ_vert[1] 
            # I_inc_ext = S_nodes[1] * (1.0 - exp(-extrap_tau / abs_μ))
            
            # Revert to standard vacuum BC for now
            RHS[1] = Irr
            
            u_bottom = S_nodes[end]
            
            A[Nnodes] = 0.0
            B[Nnodes] = 1.0
            RHS[Nnodes] = u_bottom
            
            dl .= -A[2:Nnodes]
            du .= -C[1:Nnodes-1]
            d  .= B
            
            # Solve using Julia's Tridiagonal solver (Thomas algorithm implicitly)
            tri_sol = Tridiagonal(dl, d, du)
            u_solution .= tri_sol \ RHS
            
            # Compute Diagonal Element of Inverse Operator (Lambda*) for ALI
            if !isnothing(diagonal_inv_operator)
                compute_diagonal_inv!(diag_inv_buffer, A, B, C)
                # Accumulate weighted diagonal contribution
                # Lambda_angle = w_mu * M^-1
                # J = sum w_mu * u. u = M^-1 S.
                # So Lambda_scalar = sum (w_mu * diag(M^-1)).
                @inbounds for k in 1:Nnodes
                     diagonal_inv_operator[k] += bw * (wμ * diag_inv_buffer[k])
                end
            end

            v_solution[1] = u_solution[1] - Irr
            @inbounds for k in 2:(Nnodes-1)
                dτ_minus = Δτ[k-1]
                dτ_plus  = Δτ[k]
                v_solution[k] = abs_μ * (u_solution[k+1] - u_solution[k-1]) / (dτ_minus + dτ_plus)
            end
            v_solution[end] = abs_μ * grad_S
            
            @inbounds for k in 1:Nnodes
                J_nu[k] += wμ * u_solution[k] 
                H_nu[k] += wμ * μ * v_solution[k]
            end
        end

        @inbounds for i in eachindex(J)
            F_bin = bw * (4π * H_nu[i])
            J[i] += bw * J_nu[i]
            F[i] += F_bin
            g_rad[i] += k_rho_nodes[i] / ρ[i] * F_bin / c_light
        end
    end
end







#= Parallel version ---> Needs update! =#

function _radiation_chunk_kernel(bin_range, T, ρ, z, eos, opa, 
                                μ_angles, μ_weights_scaled, bin_weights, 
                                lnrho, lnt, Δz, ncells, irradiation)
    J_nu = zeros(Float64, size(T))
    H_nu = zeros(Float64, size(T))
    
    S_nodes = zeros(Float64, size(J_nu))
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

        S_nodes .= lookup(eos, opa, :src, lnrho, lnt, bin)
        k_rho_nodes .= lookup(eos, opa, :κ, lnrho, lnt, bin)
        
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
        end
    end

    return (J_chunk, F_chunk, g_chunk)
end


"""
    update_radiation_dagger!(...)

Parallelized version of update_radiation_z_longchar! using Dagger.jl.
"""
function update_radiation_z_longchar_dagger!(J, F, g_rad; T, ρ, z, eos, opa,
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

    nbin = length(opa.λ)
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
    results = fetch.(tasks) 

    for (J_part, F_part, g_part) in results
        J .+= J_part
        F .+= F_part
        g_rad .+= g_part
    end
    
    # Enforce Monotonicity of Radiative Flux (User Requested)
    #=for i in 2:Nnodes
        if F[i] > F[i-1]
            F[i] = F[i-1]
            g_rad[i] = g_rad[i-1]
            J[i] = J[i-1]
        end
    end=#

    return nothing
end
