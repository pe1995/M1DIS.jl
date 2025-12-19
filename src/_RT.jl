function generate_mu_grid(n_points::Integer)
    μ_grid, μ_weights = gausslegendre(n_points)
    μ_grid = @. μ_grid / 2 + 0.5
    μ_weights ./= 2
    μ_grid, μ_weights
end

#=
"""
    update_radiation_z_longchar!(J, F, g_rad; T, ρ, z, eos, opa, μ_weights=TSO.labatto_4weights[1:4], μ_angles=TSO.labatto_4angles[1:4], λ_weights = nothing) 

Solve the radiative transfer equation using long characteristics. 
It is assumed that the opacities are binned, i.e. that the table contains κρ
in cm-1 and the weights already multiplied. If not, the weights can be added.
In this case one should first do TSO.@binned eos opa to emulate a binned table
and ensure that the units are correct.
"""
function update_radiation_z_longchar!(J, F, g_rad; T, ρ, z, eos, opa,
									  μ_weights=nothing,
									  μ_angles=nothing,
									  λ_weights = nothing) 
	Nnodes = length(z)
	ncells = Nnodes - 1
	Δz = diff(z)                 

	# arrays for per-bin contribution
	J_nu = zeros(size(T))
	H_nu = zeros(size(T))
    μ_angles, μ_weights = if isnothing(μ_weights)
        generate_mu_grid(5)
    else
        μ_angles, μ_weights
    end
	#scale = 0.5 / sum(μ_weights)
	#μ_weights .*= scale
    lnrho = log.(ρ)
	lnt = log.(T)

	# node-centered arrays (filled each bin)
	S_nodes = similar(J_nu)
	k_rho_nodes = similar(J_nu)
	I_up = similar(S_nodes)
	I_down = similar(S_nodes)
	
	# cell-centered arrays
	S_cell = zeros(ncells)
	k_cell = zeros(ncells)

	# reset outputs
	J .= 0.0
	F .= 0.0
	g_rad .= 0.0

	# optionally supply spectral weights
	nbin = length(opa.λ)
	bin_weights = if isnothing(λ_weights)
		ones(nbin)
	else
		λ_weights
	end

	for (bin, bw) in enumerate(bin_weights)
		S_nodes .= lookup(eos, opa, :src, lnrho, lnt, bin)
		k_rho_nodes .= lookup(eos, opa, :κ, lnrho, lnt, bin)

		@inbounds for i in 1:ncells
			S_cell[i] = 0.5 * (S_nodes[i] + S_nodes[i+1])
			k_cell[i] = 0.5 * (k_rho_nodes[i] + k_rho_nodes[i+1])
		end

		J_nu .= 0.0
		H_nu .= 0.0
		for (μ, wμ) in zip(μ_angles, μ_weights)
			# upward ray
			I_up .= 0.0
			dS = S_nodes[end] - S_nodes[end-1]
			dt = k_cell[end] * (z[end] - z[end-1])
			I_up[end] = S_nodes[end]  + (abs(μ) * dS/dt)

			@inbounds for icell in ncells:-1:1
				Δ = Δz[icell]
				κ = k_cell[icell]
				Δτ = κ * Δ / abs(μ) 
				
				trans = if Δτ < 1e-32
					1.0 - Δτ 
				else
					exp(-Δτ)
				end
				I_in = I_up[icell+1]
				S_c  = S_cell[icell]
				I_out = I_in * trans + S_c * (1 - trans)
				I_up[icell] = I_out
			end

			# downward 
			I_down .= 0.0
			I_down[1] = 0.0 

			@inbounds for icell in 1:ncells
				Δ = Δz[icell]
				κ = k_cell[icell]
				Δτ = κ * Δ / abs(μ)
				trans = if Δτ < 1e-32
					1.0 - Δτ
				else
					exp(-Δτ)
				end
				I_in = I_down[icell] 
				S_c  = S_cell[icell]
				I_out = I_in * trans + S_c * (1 - trans)
				I_down[icell+1] = I_out
			end

			@inbounds begin
				J_nu .+= wμ .* (I_up .+ I_down)
				H_nu .+= wμ .* μ .* (I_up .- I_down)
			end
		end

        @inbounds for i in eachindex(J)
		    F_bin = bw * (4π * H_nu[i])
		    #F_bin = bw * (H_nu[i])
		    J[i] += bw * J_nu[i]
            F[i] += F_bin
			g_rad[i] += k_rho_nodes[i] * F_bin / c_light
        end
	end
end
=#

# --- Helper Functions ---

function generate_mu_grid(n_points::Integer)
    x, w = gausslegendre(n_points)
    # Map from [-1, 1] to [0, 1] and scale weights
    return @. x / 2 + 0.5, @. w / 2
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
                                      λ_weights=nothing) 
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
        S_nodes .= lookup(eos, opa, :src, lnrho, lnt, bin)
        k_rho_nodes .= lookup(eos, opa, :κ, lnrho, lnt, bin)

        τ_vert[1] = 0.0 
        @inbounds for i in 1:ncells
            S_cell[i] = 0.5 * (S_nodes[i] + S_nodes[i+1])
            k_cell[i] = 0.5 * (k_rho_nodes[i] + k_rho_nodes[i+1])
            τ_vert[i+1] = τ_vert[i] + (k_cell[i] * Δz[i])
        end

        dS_bot = S_nodes[end] - S_nodes[end-1]
        dt_bot = k_cell[end] * Δz[end]
        grad_S = dt_bot > 1e-30 ? (dS_bot / dt_bot) : 0.0

        J_nu .= 0.0
        H_nu .= 0.0
        
        for (μ, wμ) in zip(use_angles, use_weights)
            abs_μ = abs(μ)

            # solve for the intensity at every node 'target_i' independently
            for target_i in 1:Nnodes
                # 1. Downward Ray (Top -> target_i)
                # Range: Cells 1 to target_i-1
                I_down = if target_i == 1
                    0.0 # Top boundary (vacuum)
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
            g_rad[i] += k_rho_nodes[i] * F_bin / c_light
        end
    end
end






#= Parallel version =#

function _radiation_chunk_kernel(bin_range, T, ρ, z, eos, opa, 
                                μ_angles, μ_weights_scaled, bin_weights, 
                                lnrho, lnt, Δz, ncells, irradiation)
    J_nu = zeros(Float64, size(T))
    H_nu = zeros(Float64, size(T))
    
    S_nodes = similar(J_nu)
    k_rho_nodes = similar(J_nu)
    I_up = similar(S_nodes)
    I_down = similar(S_nodes)
    S_cell = zeros(Float64, ncells)
    k_cell = zeros(Float64, ncells)

    J_chunk = zeros(Float64, size(T))
    F_chunk = zeros(Float64, size(T))
    g_chunk = zeros(Float64, size(T))

    for bin in bin_range
        bw = bin_weights[bin]
        Irr = isnothing(irradiation) ? 0.0 : irradiation[bin]

        S_nodes .= lookup(eos, opa, :src, lnrho, lnt, bin)
        k_rho_nodes .= lookup(eos, opa, :κ, lnrho, lnt, bin)

        @inbounds for i in 1:ncells
            S_cell[i] = 0.5 * (S_nodes[i] + S_nodes[i+1])
            k_cell[i] = 0.5 * (k_rho_nodes[i] + k_rho_nodes[i+1])
        end

        J_nu .= 0.0
        H_nu .= 0.0

        # Angular integration
        for (μ, wμ) in zip(μ_angles, μ_weights_scaled)
            # Upward ray
            dS = S_nodes[end] - S_nodes[end-1]
            dt = k_cell[end] * (z[end] - z[end-1])
			I_up[end] = S_nodes[end]  + (abs(μ) * dS/dt)

            @inbounds for icell in ncells:-1:1
                Δ = Δz[icell]
                κ = k_cell[icell]
                Δτ = κ * Δ / abs(μ)
                trans = (Δτ < 1e-32) ? (1.0 - Δτ) : exp(-Δτ)
                
                I_in = I_up[icell+1]
                S_c  = S_cell[icell]
                I_up[icell] = I_in * trans + S_c * (1 - trans)
            end

            # Downward ray
            I_down[1] = Irr # Boundary condition

            @inbounds for icell in 1:ncells
                Δ = Δz[icell]
                κ = k_cell[icell]
                Δτ = κ * Δ / abs(μ)
                trans = (Δτ < 1e-32) ? (1.0 - Δτ) : exp(-Δτ)

                I_in = I_down[icell]
                S_c  = S_cell[icell]
                I_down[icell+1] = I_in * trans + S_c * (1 - trans)
            end

            @inbounds begin
                J_nu .+= wμ .* (I_up .+ I_down)
                H_nu .+= wμ .* μ .* (I_up .- I_down)
            end
        end

        @inbounds for i in eachindex(J_chunk)
            F_bin = bw * (4π * H_nu[i])
            #F_bin = bw * (H_nu[i])
            J_chunk[i] += bw * J_nu[i]
            F_chunk[i] += F_bin
            g_chunk[i] += k_rho_nodes[i] * F_bin / c_light 
        end
    end

    return (J_chunk, F_chunk, g_chunk)
end


"""
    update_radiation_dagger!(...)

Parallelized version of update_radiation_z_longchar! using Dagger.jl.
"""
function update_radiation_z_longchar_dagger!(J, F, g_rad; T, ρ, z, eos, opa,
                                  μ_weights=TSO.labatto_4weights[1:4],
                                  μ_angles=TSO.labatto_4angles[1:4],
                                  λ_weights=nothing, irradiation=nothing) 
    
    Nnodes = length(z)
    ncells = Nnodes - 1
    Δz = diff(z)
    
    lnrho = log.(ρ)
    lnt = log.(T)

    scale = 0.5 / sum(μ_weights)
    μ_weights_scaled = μ_weights .* scale

    nbin = length(opa.λ)
    bin_weights = isnothing(λ_weights) ? ones(nbin) : λ_weights

    n_workers = Base.Threads.nthreads() 
    chunk_size = max(1, cld(nbin, n_workers))
    chunks = Iterators.partition(1:nbin, chunk_size)

    tasks = map(chunks) do range
        Dagger.@spawn _radiation_chunk_kernel(
            range, T, ρ, z, eos, opa, 
            μ_angles, μ_weights_scaled, bin_weights, 
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

    return nothing
end