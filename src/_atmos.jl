# ============================================================================
# Iterative computation of the Atmosphere 
# ============================================================================

"""
    atmosphere(; T_eff, logg, eos, opacity, kwargs...)

Compute a M1DIS atmosphere iteratively based on the given binned opacity table, effective temperature and surface gravity.

# Arguments
- `T_eff`: effective temperature
- `logg`: surface gravity
- `eos`: equation of state
- `opacity`: opacity table
- `τ`: optical depth grid
- `α_MLT`: mixing length parameter
- `maxiter`: maximum number of iterations
- `damping`: damping parameter for limiting dT correction
- `v_mac`: microturbulent velocity to add artificial turbulent pressure
- `T_irradiation`: irradiation temperature
- `R_irradiation`: irradiation radius
- `d_irradiation`: irradiation distance
- `F_irradiation`: irradiation flux
- `T`: temperature (optional starting atmosphere)
- `ρ`: density (optional starting atmosphere)
- `P`: pressure (optional starting atmosphere)
- `z`: height (optional starting atmosphere)
- `feutrier`: use Feutrier solver
- `use_threads`: use threads
- `dt_tolerance_rel`: relative temperature tolerance
- `flux_tolerance_rel`: relative flux tolerance
- `save_every`: save every n iterations
- `kwargs...`: additional keyword arguments

# Returns
- `result`: M1DIS.jl atmosphere or array of M1DIS.jl atmospheres (in MUST.Box format consistent with DISPATCH).
"""
function atmosphere(; T_eff, logg, eos, opacity, 
	τ=10 .^range(-6.0, 2.0, length=100), 
	α_MLT=1.5, 
	maxiter=20,
	damping=0.1, 
	v_mac=0.0,
	T_irradiation=nothing, R_irradiation=nothing, d_irradiation=nothing, F_irradiation=nothing,
	T=nothing, ρ=nothing, P=nothing, z=nothing, 
	feutrier=true,
	use_threads=false,
	scattering_opacity=nothing,
	kwargs...)	

	@optionalTiming initialization_time begin
		# input opacities and EoS
		eos = if typeof(eos) <: TSO.ExtendedEoS
			@assert !TSO.is_internal_energy(eos.eos)
			eos
		else
			@assert !TSO.is_internal_energy(@axed(eos))
			eos = TSO.extended(eos)
			TSO.add_thermodynamics!(eos)

			eos
		end
		
		opacity = if typeof(opacity) <: TSO.SqOpacity
			o = TSO.ExtendedOpacity(opa=opacity)
			@verbose_warn 2 "Opacity was passed without wrapping it into an ExtendedOpacity. This is not recommended, as it will be guessed if the table is binned or not."
			@verbose_warn 2 "The guess is: $(o.binned ? "binned" : "not binned")"
			@verbose_warn 2 "If this is not true, please pass the opacity as: `TSO.ExtendedOpacity(opa=opaccity, binned=binned)`"
			o
		else
			opacity
		end
		
		@assert typeof(opacity) <: Union{TSO.ExtendedOpacity, TSO.MiniOpacityTable} "Opacity must be an ExtendedOpacity or MiniOpacityTable."
		
		if opacity.binned
			@assert typeof(opacity) <: TSO.ExtendedOpacity "Binned opacities must use ExtendedOpacity to provide the source function. MiniOpacityTable cannot be binned."
		end
		
		if !opacity.binned
			use_threads = true
		end

		if opacity.binned && (typeof(opacity) <: TSO.ExtendedOpacity)
			if !haskey(opacity.extensions, :dS_dT)
				TSO.gradients!(TSO.table(eos), opacity)
			end
		end

		# scattering opacity
		if !isnothing(scattering_opacity)
			if !(typeof(scattering_opacity) <: TSO.MiniOpacityTable)
				error("Scattering opacity must be of type MiniOpacityTable. $(typeof(scattering_opacity))")
			end
			@info "Running with scattering treatment."
		end

		# if no atmosphere is provided, compute an initial gray atmosphere
		τ = deepcopy(τ)
		T, ρ, P, z = if isnothing(T) 
			initial_atmosphere(τ, T_eff=T_eff, logg=logg, eos=eos)
		else
			deepcopy(T), deepcopy(ρ), deepcopy(P), deepcopy(z)
		end
		
		J, F_rad, g_rad, P_rad = fill!(similar(T), 0.0), fill!(similar(T), 0.0), fill!(similar(T), 0.0), fill!(similar(T), 0.0)
		F_conv, v_conv, g_turb, P_turb = fill!(similar(T), 0.0), fill!(similar(T), 0.0), fill!(similar(T), 0.0), fill!(similar(T), 0.0)
		dFconv_dT = fill!(similar(T), 0.0)
		dT = fill!(similar(T), 0.0)
		dT_rel = fill!(similar(T), 0.0)
		dT_rel_smoothed = fill!(similar(T), 0.0)
		F_target = σ_SB * T_eff^4
		#F_conv_old = fill!(similar(T), 0.0)
		#dFconv_dT_old = fill!(similar(T), 0.0)
		F_total = fill!(similar(T), 0.0)
		F_err_rel = fill!(similar(T), 0.0)
		μ_angles, μ_weights = generate_mu_grid(4)

		# check for irradiation and compute it
		Irr = isnothing(d_irradiation) ? nothing : irradiate(eos, opacity, T_irradiation, R_irradiation, d_irradiation, F_irradiation)

		# initialize the Feutrier RT solver storage arrays
		chi, chi_ref, S, dSdT, chi_scat, atm = if feutrier
			@optionalTiming prepare_opacities_time begin
				chi, chi_ref, S, dSdT = if opacity.binned
					compute_opacities(eos, opacity, T, ρ)
				else
					compute_opacities_chunked(eos, opacity, T, ρ)
				end

				chi_scat, _, _, _ = if !isnothing(scattering_opacity)
					compute_opacities_chunked(eos, scattering_opacity, T, ρ, opacity_only=true)
				else
					nothing, nothing, nothing, nothing
				end
			end
			@optionalTiming allocate_feutrier_time begin
				atm = Atmosphere(
					T_eff=T_eff, z=z, 
					tau=τ, rho=ρ, Temp=T, 
					F_conv=F_conv, dFconv_dT=dFconv_dT,
					mu=μ_angles, w_mu=μ_weights, 
					chi=chi, chi_ref=chi_ref, B=S, dBdT=dSdT, I_top=Irr, chi_scat=chi_scat
				)
			end
			chi, chi_ref, S, dSdT, chi_scat, atm
		else
			nothing, nothing, nothing, nothing, nothing
		end
		
		@verbose_info 2 "================================= M1DIS ================================="
		
		r = []
		if (typeof(opacity) <: TSO.MiniOpacityTable)
			@verbose_info 2 """
			Running M1DIS with MiniOpacityTable. 
			This causes the source function and its derivative to be 
			computed on the fly to save memory.
			This means the source function is always assumed to be the Planck function.
			"""
		end
		
		if !opacity.binned
			@verbose_info 2 """
			Running M1DIS with unbinned opacity table.
			Forcing `use_threads=true` to handle the large number of frequency points.
			"""
		end

		if use_threads
			@verbose_info 1 "Running approximate Feutrier RT with $(Base.Threads.nthreads()) threads."
		end

		@verbose_info 2 "========================================================================="
		@verbose_info 1 "iteration | relative flux error (max) | relative T error (max) | ΔT (max)" 
	end
	flux_err_max_prev = Inf
	base_damping = damping
	@optionalTiming relaxation_time for iter in 1:maxiter
        #P_turb_old .= P_turb
		#F_conv_old .= F_conv
		#dFconv_dT_old .= dFconv_dT
        
		# compute convective quantities (MLT)
		@optionalTiming mixing_length_time update_mixing_length!(F_conv, v_conv, P_rad, P_turb, dFconv_dT, T, P, ρ, τ, eos, exp10(logg); alpha_mlt=α_MLT, Teff=T_eff, v_mac=v_mac*1e5)
		
        #if iter > 1
        #    P_turb .= 0.5 .* P_turb_old .+ 0.5 .* P_turb
        #    F_conv .= 0.5 .* F_conv_old .+ 0.5 .* F_conv
        #    dFconv_dT .= 0.5 .* dFconv_dT_old .+ 0.5 .* dFconv_dT
        #end

		# Monotonic Enforcer during early relaxation (Error > 10%)
		#=damping = if flux_err_max_prev > 50.0
			for n in 2:length(F_conv)
				if (F_conv[n-1] > 0.0) && (F_conv[n] < F_conv[n-1])
					F_conv[n] = F_conv[n-1]
					v_conv[n] = v_conv[n-1]
					P_turb[n] = P_turb[n-1]
					dFconv_dT[n] = dFconv_dT[n-1]
				end
			end
			fconv_stabilizer!(F_conv, passes=1)
			fconv_stabilizer!(v_conv, passes=1)
			fconv_stabilizer!(P_turb, passes=1)
			fconv_stabilizer!(dFconv_dT, passes=1)
			base_damping
		elseif flux_err_max_prev > 1.0
			fconv_stabilizer!(F_conv, passes=1)
			fconv_stabilizer!(v_conv, passes=1)
			fconv_stabilizer!(P_turb, passes=1)
			fconv_stabilizer!(dFconv_dT, passes=1)
			base_damping * 0.5
		else
			base_damping * 0.1
		end=#
		
        
		@optionalTiming radiation_transfer_time begin
			if opacity.binned
 				@optionalTiming compute_opacities_time compute_opacities!(chi, chi_ref, S, dSdT, eos, opacity, T, ρ)
			else
 				@optionalTiming compute_opacities_time compute_opacities_chunked!(chi, chi_ref, S, dSdT, eos, opacity, T, ρ)
			end

			if !isnothing(scattering_opacity)
				@optionalTiming compute_opacities_time compute_opacities_chunked!(chi_scat, nothing, nothing, nothing, eos, scattering_opacity, T, ρ)
			end

			@optionalTiming update_atmosphere_time update!(atm; tau=τ, rho=ρ, Temp=T, F_conv=F_conv, dFconv_dT=dFconv_dT, chi=chi, chi_ref=chi_ref, B=S, dBdT=dSdT, chi_scat=chi_scat)
			
			@optionalTiming solve_RT_time if !use_threads
				solve_gustafsson!(atm, include_dT=true)
			else
				solve_approximate!(atm)
			end

			J .= atm.J_bol
			F_rad .= atm.F_bol
			g_rad .= atm.g_rad
			P_rad .= atm.P_rad
			dT .= atm.dT

			# Update tracking for MLT stabilization
            F_total .= atm.F_bol .+ atm.F_conv
			F_err_rel .= (F_total .- F_target) ./ F_target
            flux_err_max = maximum(abs.(F_err_rel))

			#damping = if flux_err_max > flux_err_max_prev
			#	max(damping * 0.75, 0.001)
			#end
			flux_err_max_prev = flux_err_max

			#=dT_rel .= dT ./ T
			dT_rel_smoothed .= dT_rel
			for i in 2:length(dT_rel)-1
				dT_rel_smoothed[i] = 0.25 * dT_rel[i-1] + 0.5 * dT_rel[i] + 0.25 * dT_rel[i+1]
			end
			dT .= dT_rel_smoothed .* T=#
			dT .= clamp.(dT, -damping.*T, damping.*T)
		end

		converged, new_damping = evaluate_iteration!(
			r, iter, maxiter, F_target, dT, τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, T_eff, logg, eos, damping; 
			dFconv_dT=dFconv_dT, J=J, g_turb=g_turb, g_rad=g_rad, P_turb=P_turb, P_rad=P_rad, F_err_rel=F_err_rel, 
			chi_max=maximum(chi, dims=1), chi_min=minimum(chi, dims=1), #chi_scat_max=maximum(chi_scat, dims=1), chi_scat_min=minimum(chi_scat, dims=1),
			kwargs...
		)

		if new_damping != damping
			@verbose_info 2 "Increasing damping to $(new_damping) (from $(damping))."
			damping = new_damping
		end

		if converged 
			@verbose_info 1 "Atmosphere converged."
			break
		end

		T .+= dT
		T = clamp.(T, 10, 1e12)
		#force_adiabatic_bottom!(T, P, eos, n_force=2)
		@optionalTiming hydrostatic_time update_hydrostatic!(P, ρ, z, T, P_turb, P_rad, τ, eos=eos, logg=logg)
	end

    length(r) == 1 ? r[1] : r
end

# ============================================================================
# Initial Atmosphere
# ============================================================================

function initial_atmosphere(τ_grid; T_eff, logg, eos)
    # Gray atmosphere
	T_initial = T_eff .* (0.75 * (τ_grid .+ 2/3)) .^ 0.25

	ρ_initial = similar(T_initial)
	P_initial = similar(T_initial)
	z_initial = similar(T_initial)
	#g_rad = similar(T_initial)
	#g_turb = similar(T_initial)
	P_rad = similar(T_initial)
	P_turb = similar(T_initial)

	#g_rad .= 0.0
	#g_turb .= 0.0
	P_rad .= 0.0
	P_turb .= 0.0
	update_hydrostatic!(
		P_initial, ρ_initial, z_initial, T_initial, P_turb, P_rad, τ_grid, 
		logg=logg, eos=eos
	)

	T_initial, ρ_initial, P_initial, z_initial
end

# ============================================================================
# Iteration Evaluation
# ============================================================================

function evaluate_iteration!(result, 
	iter, maxiter, 
	F_target, dT, 
	τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, teff, logg, eos, damping; 
	dt_tolerance_rel=0.0001, flux_tolerance_rel=0.01, save_every=1, kwargs...)
	# store the atmosphere every `save_every` iterations
	store = save_every > 0 ? ((iter%save_every == 0) | (iter == maxiter)) : false
    F_total = F_rad .+ F_conv
	flux_err_max = maximum(abs.(F_total[2:end-1] .- F_target)) / F_target
	dt_err_max = maximum(abs.(dT[2:end-1] ./ T[2:end-1]))
	sinf = TSO.@sprintf("%4d | %16.4f | %14.4f | %10.1f K\n", 
			iter, flux_err_max*100, dt_err_max*100, maximum(abs.(dT[2:end-1])))
	@verbose_info 1 sinf

	new_damping = damping #flux_err_max < 0.05 ? min(damping * 1.05, 0.25) : damping

	converged = (dt_err_max<dt_tolerance_rel) | (flux_err_max<flux_tolerance_rel)
	if converged | store
		append!(result, [m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos; kwargs...)])
	end

	converged, new_damping
end

# ============================================================================
# Return the M1DIS.jl result in the same format as in DISPATCH
# ============================================================================

m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos; kwargs...) = begin
    p = MUST.AtmosphericParameters(-99.0, Base.convert(Float64, teff), Base.convert(Float64, logg), Dict{Symbol,Float64}())
    zz = reshape(z, 1, 1, :) |> deepcopy
    xx = zeros(size(zz))
    yy = zeros(size(zz))

	τ_new = deepcopy(τ)
	#update_τ_grid!(τ_new; T=T, ρ=ρ, z=z, eos=eos.eos)

    d = Dict(
        :τ_ross=>reshape(τ_new, 1, 1, :) |> deepcopy,
        :T=>reshape(T, 1, 1, :) |> deepcopy,
        :d=>reshape(ρ, 1, 1, :) |> deepcopy,
        :Pg=>reshape(P, 1, 1, :) |> deepcopy,
        :F_rad=>reshape(F_rad, 1, 1, :) |> deepcopy,
        :F_conv=>reshape(F_conv, 1, 1, :) |> deepcopy,
        :dFconv_dT=>reshape(dFconv_dT, 1, 1, :) |> deepcopy,
		:dT=>reshape(dT, 1, 1, :) |> deepcopy,
    )

	for (k, v) in kwargs
		d[k] = reshape(v, 1, 1, :) |> deepcopy
	end

    MUST.Box(xx, yy, zz, d, p)
end

# ============================================================================
# Saving the M1DIS.jl result in the Multi3D and Multi1D format
# ============================================================================

function save!(model_data::MUST.Box, model_name; eos500=nothing, folder="./", vmic=0.0, logg=4.5)
    base_path = abspath(folder)
    
    if !isdir(base_path)
        mkpath(base_path)
    end
    
    run_i = joinpath(base_path, model_name)
    if !isdir(run_i)
        mkdir(run_i)
    end

    model_data.z .= model_data.z .- TSO.optical_surface(model_data.data[:τ_ross][1,1,:], model_data.z[1,1,:])
    MUST.flip!(model_data, depth=true)

    b = model_data
    z = b[:z][1,1,:]
    rho = b[:d][1,1,:]
    T = b[:T][1,1,:]
    vmic_arr = fill(vmic, length(z))
    
    # 1. M3D format
    f_new = joinpath(run_i, "$(model_name)_m3d.txt")
    MUST.save_text_m3d(f_new, z, rho, T; header=model_name, vmic=vmic_arr)
    
    # 2. M1D format
	if !isnothing(eos500)
    	MUST.flip!(model_data, depth=false)
		model_data.data[:Ne] = reshape(TSO.lookup(eos500, :lnNe, log.(model_data[:d]), log.(model_data[:T])) .|> exp, 1, 1, :)
		model_data.data[:κ500] = reshape(TSO.lookup(eos500, :lnRoss, log.(model_data[:d]), log.(model_data[:T])) .|> exp, 1, 1, :)
    	model_data.data[:τ500] = MUST.optical_depth(model_data, opacity=:κ500, density=:d)
    	MUST.flip!(model_data, depth=true)

		tau500 =log10.(b[:τ500][1,1,:])
    	Ne = b[:Ne][1,1,:]

		f_new_m1d = joinpath(run_i, "atmos.$(model_name)")
		MUST.save_text_m1d(f_new_m1d, tau500, T, Ne; logg=logg, header=model_name, vmic=vmic_arr)
		
		f_new_dscale = joinpath(run_i, "dscale.$(model_name)")
		MUST.save_text_m1d_dscale(f_new_dscale, tau500; header=model_name)
	end

	# 3. HDF5 format
	MUST.flip!(model_data)
	MUST.save(model_data, folder=run_i, name=model_name)

    return run_i
end



