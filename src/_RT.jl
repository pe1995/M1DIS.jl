"""
    update_radiation_z_longchar!(J, F, g_rad; T, ρ, z, eos, opa, μ_weights=TSO.labatto_4weights[1:4], μ_angles=TSO.labatto_4angles[1:4], λ_weights = nothing) 

Solve the radiative transfer equation using long characteristics. 
It is assumed that the opacities are binned, i.e. that the table contains κρ
in cm-1 and the weights already multiplied. If not, the weights can be added.
In this case one should first do TSO.@binned eos opa to emulate a binned table
and ensure that the units are correct.
"""
function update_radiation_z_longchar!(J, F, g_rad; T, ρ, z, eos, opa,
									  μ_weights=TSO.labatto_4weights[1:4],
									  μ_angles=TSO.labatto_4angles[1:4],
									  λ_weights = nothing) 
	Nnodes = length(z)
	ncells = Nnodes - 1
	Δz = diff(z)                 

	# arrays for per-bin contribution
	J_nu = zeros(size(T))
	H_nu = zeros(size(T))
	scale = 0.5 / sum(μ_weights)
	μ_weights .*= scale
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
			I_up[end] = S_nodes[end]  + (abs(μ) * dS)

			@inbounds for icell in ncells:-1:1
				Δ = Δz[icell]
				κ = k_cell[icell]
				Δτ = κ * Δ / abs(μ) 
				
				trans = if Δτ < 1e-6
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
				trans = if Δτ < 1e-6
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
		    J[i] += bw * J_nu[i]
            F[i] += F_bin
			g_rad[i] += k_rho_nodes[i] * F_bin / c_light
        end
	end
end







#= Parallel version =#

function _radiation_chunk_kernel(bin_range, T, ρ, z, eos, opa, 
                                μ_angles, μ_weights_scaled, bin_weights, 
                                lnrho, lnt, Δz, ncells)
    J_nu = zeros(eltype(T), size(T))
    H_nu = zeros(eltype(T), size(T))
    
    S_nodes = similar(J_nu)
    k_rho_nodes = similar(J_nu)
    I_up = similar(S_nodes)
    I_down = similar(S_nodes)
    S_cell = zeros(eltype(T), ncells)
    k_cell = zeros(eltype(T), ncells)

    J_chunk = zeros(eltype(T), size(T))
    F_chunk = zeros(eltype(T), size(T))
    g_chunk = zeros(eltype(T), size(T))

    for bin in bin_range
        bw = bin_weights[bin]

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
            I_up .= 0.0
            dS = S_nodes[end] - S_nodes[end-1]
            I_up[end] = S_nodes[end] + (abs(μ) * dS)

            @inbounds for icell in ncells:-1:1
                Δ = Δz[icell]
                κ = k_cell[icell]
                Δτ = κ * Δ / abs(μ)
                trans = (Δτ < 1e-6) ? (1.0 - Δτ) : exp(-Δτ)
                
                I_in = I_up[icell+1]
                S_c  = S_cell[icell]
                I_up[icell] = I_in * trans + S_c * (1 - trans)
            end

            # Downward ray
            I_down .= 0.0
            I_down[1] = 0.0 # Boundary condition

            @inbounds for icell in 1:ncells
                Δ = Δz[icell]
                κ = k_cell[icell]
                Δτ = κ * Δ / abs(μ)
                trans = (Δτ < 1e-6) ? (1.0 - Δτ) : exp(-Δτ)

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
                                  λ_weights=nothing) 
    
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
            lnrho, lnt, Δz, ncells
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