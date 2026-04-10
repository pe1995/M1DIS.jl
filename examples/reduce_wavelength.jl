using Pkg; Pkg.activate(".")
using M1DIS
using MUST 
using TSO
using Random
using StatsBase

function reduced_opacity(opa, indices)
	opa_core = TSO.opacity(opa)
	k_cut = opa_core.κ[:, :, indices]
	k_lambda = opa_core.λ[indices]
	s_cut = if size(opa_core.src, 3) == size(opa_core.κ, 3)
		opa_core.src[:, :, indices]
	else
		copy(opa_core.src)
	end

	opa_new = TSO.RegularOpacityTable(
		k_cut, 
		copy(opa_core.κ_ross), 
		s_cut, 
		k_lambda, 
		false
	)
	opa_new_type = if typeof(opa) <: TSO.ExtendedOpacity
		opa_new |> extended
	else
		TSO.MiniOpacityTable(opa_new, false, TSO.ω_midpoint(opa_new))
	end

	opa_new_type
end

function reduced_opacity!(opa_reduced, opa, indices)
	opa_core = TSO.opacity(opa)
	opacity(opa_reduced).κ .= opa_core.κ[:, :, indices]
	opacity(opa_reduced).λ .= opa_core.λ[indices]
	s_cut = if size(opa_core.src, 3) == size(opa_core.κ, 3)
		@view opa_core.src[:, :, indices]
	else
		opa_core.src
	end
	opacity(opa_reduced).src .= s_cut
	opa_reduced.weights .= TSO.ω_midpoint(opa_reduced |> opacity)

	opa_reduced
end

function run_1d_rt!(atm, eos, opa)
    M1DIS.compute_opacities_chunked!(
		atm.chi, 
		atm.chi_ref, 
		atm.B, 
		atm.dBdT, 
		atm.dchidT, 
		eos, 
		opa, 
		atm.Temp, 
		atm.rho
	)
    M1DIS.update!(atm)
    M1DIS.solve_approximate!(atm, include_dT=false)
	
	atm
end

function optimize_grid_multi_atm(
    atmospheres, 
    eos, opa,
    target_points::Int=10000; 
    max_iter::Int=100, 
    save_interval::Int=50,    
	save_name = "lambda_opt_feh0"
)
    num_atms = length(atmospheres)
    full_indices = eachindex(wavelength(opa))
    total_points = length(full_indices)
	cooling_rate = exp(log(0.01) / max_iter)
    
    istep = floor(Int, total_points / target_points)
    selected_indices = full_indices[1:istep:end]
    selected_indices = selected_indices[1:target_points] |> collect

    in_grid = falses(total_points)
    in_grid[selected_indices] .= true

    opa_reduced = reduced_opacity(opa, selected_indices)
    atm_red = [M1DIS.Atmosphere(m, eos, opa_reduced) for m in atmospheres]
    atm_full = [M1DIS.Atmosphere(m, eos, opa) for m in atmospheres]
    
    flux_true = []
    Q_true = []
    for (i, atm) in enumerate(atm_full)
        run_1d_rt!(atm, eos, opa)
        push!(flux_true, copy(atm.F_rad))
        push!(Q_true, copy(atm.Q_rad))
    end

    @info("Building Physics-Informed Prior from high-res opacity table...")
    add_weights = zeros(Float64, total_points)
    
    for atm in atm_full
        log_chi = log10.(max.(atm.chi, 1e-30)) 
        chi_grad = abs.(diff(log_chi, dims=1))
        
        max_grad_per_lambda = maximum(chi_grad, dims=2)[:] 
        push!(max_grad_per_lambda, 0.0)
        
        add_weights .+= max_grad_per_lambda
    end
    
    add_weights ./= maximum(add_weights)
    add_weights .+= 0.05 
    add_weights[selected_indices] .= 0.0     
    sampling_weights = Weights(add_weights)
    epsilon = 1e-15 
    Q_norm = [maximum(abs, q) for q in Q_true]

    compute_total_loss(indices) = begin
        total_loss = 0.0
        reduced_opacity!(opa_reduced, opa, indices)
        
        for (i, atm) in enumerate(atm_red)
            run_1d_rt!(atm, eos, opa_reduced)
            ff = flux_true[i]
            qq = Q_true[i]
            qn = Q_norm[i]
            
            @inbounds for j in eachindex(atm.F_rad)
                rel_err_F = (atm.F_rad[j] - ff[j]) / (abs(ff[j]) + epsilon)
                rel_err_Q = (atm.Q_rad[j] - qq[j]) / (qn + epsilon)
                
                total_loss += rel_err_F^2 + rel_err_Q^2
            end            
        end
        return total_loss
    end

    @info("Evaluating initial baseline grid...")
    current_loss = compute_total_loss(selected_indices)
    best_indices = copy(selected_indices)
    temp_indices = copy(selected_indices)
    best_loss = current_loss

    @info("Setting initial temperature based on baseline loss...")
    temp = (0.01 * current_loss) / 0.693
	
    @info("Starting optimization (Initial Loss: $(round(current_loss, sigdigits=5)), Temp: $(round(temp, sigdigits=4)))")
    
    for iter in 1:max_iter
        drop_pos = rand(1:target_points)
        val_to_drop = selected_indices[drop_pos]
        
        # Pick the new point using the Opacity Prior / Learned Weights
        val_to_add = StatsBase.sample(1:total_points, sampling_weights)
        
        temp_indices .= selected_indices
        temp_indices[drop_pos] = val_to_add
        sort!(temp_indices)
        
        new_loss = compute_total_loss(temp_indices)
        loss_diff = new_loss - current_loss
        
        if loss_diff < 0 || rand() < exp(-loss_diff / temp)
            selected_indices .= temp_indices
            current_loss = new_loss
            
            in_grid[val_to_drop] = false
            in_grid[val_to_add] = true
            
            if loss_diff < 0
                idx_min = max(1, val_to_add - 5)
                idx_max = min(total_points, val_to_add + 5)
                add_weights[idx_min:idx_max] .*= 1.1 
            end
            
            add_weights[val_to_add] = 0.0 
            add_weights[val_to_drop] = 0.05 
            sampling_weights = Weights(add_weights)
            
            if current_loss < best_loss
                best_loss = current_loss
                best_indices .= selected_indices
            end
        end
        temp *= cooling_rate
        
        N_evaluations = num_atms * length(atm_full[1].F_rad) * 2 # *2 for F and Q
        rms_percent = sqrt(current_loss / N_evaluations) * 100
        best_rms_percent = sqrt(best_loss / N_evaluations) * 100

        if iter % 1 == 0
            @info(
				"[Iter $iter / $max_iter] | RMS: $(round(rms_percent, sigdigits=5))% | Best RMS: $(round(best_rms_percent, sigdigits=5))% | Temp: $(round(temp, sigdigits=4))"
			)
        end

		if iter % save_interval == 0
			open(save_name*"_$(iter).txt", "w") do fio
				M1DIS.writedlm(fio, [best_indices wavelength(opa)[best_indices]])
			end
		end
    end

    @info("Optimization complete! Final Best Loss: $best_loss")
    
	open(save_name*"_final.txt", "w") do fio
		M1DIS.writedlm(fio, [best_indices wavelength(opa)[best_indices]])
	end
    
    return best_indices
end

# ============================================================================
# Input
# ============================================================================
begin
    eos_dir = "../../opacity_tables/magg_m0_a0_vmic1_v3.5/"
    models = [
        Box("../models/p5777.0_g4.44_z0.0_a0.0_vmic1.0/p5777.0_g4.44_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
        Box("../models/p5000.0_g5.0_z0.0_a0.0_vmic1.0/p5000.0_g5.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p7000.0_g5.0_z0.0_a0.0_vmic1.0/p7000.0_g5.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p5000.0_g4.0_z0.0_a0.0_vmic1.0/p5000.0_g4.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p7000.0_g4.0_z0.0_a0.0_vmic1.0/p7000.0_g4.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p4000.0_g3.0_z0.0_a0.0_vmic1.0/p4000.0_g3.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p6000.0_g3.0_z0.0_a0.0_vmic1.0/p6000.0_g3.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p4000.0_g2.0_z0.0_a0.0_vmic1.0/p4000.0_g2.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p6000.0_g2.0_z0.0_a0.0_vmic1.0/p6000.0_g2.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		#Box("../models/p4000.0_g1.5_z0.0_a0.0_vmic1.0/p4000.0_g1.5_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		#Box("../models/p6000.0_g1.5_z0.0_a0.0_vmic1.0/p6000.0_g1.5_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p4000.0_g1.0_z0.0_a0.0_vmic1.0/p4000.0_g1.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false),
		Box("../models/p5000.0_g1.0_z0.0_a0.0_vmic1.0/p5000.0_g1.0_z0.0_a0.0_vmic1.0.hdf5", mmap=false)
    ]

    out_name = "lambda10597_feh0"
    rm(out_name*"_*.txt", force=true)
    n_points = 10597
    max_iter = 50000
    save_interval = floor(Int, max_iter/10)
end

# ============================================================================
# Load Data
# ============================================================================
begin
	eos_file = MUST.glob("*_eos_*.hdf5", eos_dir)[1]
	eos500_file = MUST.glob("*_eos500_*.hdf5", eos_dir)[1]
	opa_file = MUST.glob("*_opacities_*.hdf5", eos_dir)[1]
	scat_file = MUST.glob("*_sopacities_*.hdf5", eos_dir)[1]
	
	eos = reload(eos_file) |> extended
	opa = reload(opa_file, mmap=true) |> extended
end

# ============================================================================
# Run Optimization 
# ============================================================================
begin
    test = optimize_grid_multi_atm(
        models, 
        eos, 
        opa, 
        n_points, 
        max_iter=max_iter, 
        save_interval=save_interval, 
        save_name=out_name
    )
end

