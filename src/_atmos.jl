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

function initial_atmosphere(τ_grid; T_eff, logg, eos)
    # Gray atmosphere
	T_initial = T_eff .* (0.75 * (τ_grid .+ 2/3)) .^ 0.25

	ρ_initial = similar(T_initial)
	P_initial = similar(T_initial)
	z_initial = similar(T_initial)
	g_rad = similar(T_initial)
	g_turb = similar(T_initial)

	g_rad .= 0.0
	g_turb .= 0.0
	update_hydrostatic!(
		P_initial, ρ_initial, z_initial, T_initial, g_rad, g_turb, τ_grid, 
		logg=logg, eos=eos
	)

	T_initial, ρ_initial, P_initial, z_initial
end





#= Iterative computation of the Atmosphere =#

function evaluate_iteration!(result, iter, maxiter, F_target, dT, τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, teff, logg, eos; dt_tolerance_rel=0.001, flux_tolerance_rel=0.001, save_every=-1, kwargs...)
	store = save_every > 0 ? ((iter%save_every == 0) | (iter == maxiter)) : false
    F_total = F_rad .+ F_conv
	flux_err_max = maximum(abs.(F_total[2:end-1] .- F_target)) / F_target
	dt_err_max = maximum(abs.(dT[2:end-1] ./ T[2:end-1]))
	sinf = TSO.@sprintf("%4d | %16.4f | %14.4f | %10.1f K\n", 
			iter, flux_err_max*100, dt_err_max*100, maximum(abs.(dT[2:end-1])))
	@info sinf

	converged = (dt_err_max<dt_tolerance_rel) | (flux_err_max<flux_tolerance_rel)
	if converged | store
		append!(result, [m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos; kwargs...)])
	end

	converged
end

"""
    atmosphere(; T_eff, logg, eospath, τ=10 .^range(-5.0, 4, length=100), α_MLT=1.5, maxiter=200)	

Compute a M1DIS atmosphere iteratively based on the given binned opacity table, effective temperature and surface gravity.
"""
function atmosphere(; T_eff, logg, eos, opacity, 
	τ=10 .^range(-5.0, 4, length=100), 
	α_MLT=1.5, 
	maxiter=500,
	damping=0.4, 
	λ_weights=nothing, 
	T_irradiation=nothing, R_irradiation=nothing, d_irradiation=nothing, 
	use_threads=false, 
	mafags_mlt=false, 
	feutrier=false, 
	T=nothing, ρ=nothing, P=nothing, z=nothing, 
	kwargs...)	

	eos = if typeof(eos) <: TSO.ExtendedEoS
		@assert !TSO.is_internal_energy(@axed(eos.eos))
		eos
	else
		@assert !TSO.is_internal_energy(@axed(eos))
		eos = TSO.ExtendedEoS(eos=eos)
		TSO.add_thermodynamics!(eos)

		eos
	end
	
	opa, λ_weights = if typeof(opacity) <: TSO.BinnedOpacities
		w = if isnothing(λ_weights)
			@warn "You passed a binned opacity object. If you are doing this because the table you are using is not actually binned, remember to pass λ_weights! Assuming midpoint from table."
			TSO.ω_midpoint(opacity.opacities)
		else
			λ_weights
		end
		opacity.opacities, w
	else
		opacity, ones(length(opacity.λ))
	end

	opa = TSO.ExtendedOpacity(opa=opa)
	TSO.gradients!(eos.eos, opa)

	τ = deepcopy(τ)
	T, ρ, P, z = if isnothing(T) 
		initial_atmosphere(τ, T_eff=T_eff, logg=logg, eos=eos)
	else
		deepcopy(T), deepcopy(ρ), deepcopy(P), deepcopy(z)
	end
	J, F_rad, g_rad = similar(T), similar(T), similar(T)
	J_plus, F_rad_plus, g_rad_plus = similar(T), similar(T), similar(T)
	J_minus, F_rad_minus, g_rad_minus = similar(T), similar(T), similar(T)
	Jmat = zeros(size(J, 1), size(J, 1))
	F_conv, v_conv, g_turb = similar(T), similar(T), similar(T)
	F_conv_plus, F_conv_minus = similar(T), similar(T)
	Q, dQdT = similar(T), similar(T)
    dFconv_dT = similar(T) 
    dFrad_dT = similar(T) 
    dFconv_dT_minus = similar(T) 
	dT = similar(T)
	smalldT = similar(T)
    lambda_diagonal = similar(T)
    F_target = σ_SB * T_eff^4

    μ_angles, μ_weights = generate_mu_grid(4)
    

	# check for irradiation and compute it
	Irr = isnothing(T_irradiation) ? nothing : irradiate(eos, opa.opa, T_irradiation, R_irradiation, d_irradiation)

    @info "============================== M1DIS ===================================="
	@info "iteration | relative flux error (max) | relative T error (max) | ΔT (max)" 
	
	r = []
	if use_threads
		@info "Running RT with $(Base.Threads.nthreads()) threads."
	end
	for iter in 1:maxiter
		#=if use_threads
			update_radiation_z_longchar_dagger!(
				J, F_rad, g_rad, Q, dQdT, T=T, ρ=ρ, z=z, eos=eos.eos, opa=opa, λ_weights=λ_weights, irradiation=Irr, μ_weights=μ_weights, μ_angles=μ_angles
			)
		else
			if !feutrier
				update_radiation_z_longchar!(
					J, F_rad, g_rad, T=T, ρ=ρ, z=z, eos=eos.eos, opa=opa.opa, λ_weights=λ_weights, irradiation=Irr, μ_weights=μ_weights, μ_angles=μ_angles
				)
			else
				update_radiation_z_feutrier!(
					J, F_rad, g_rad, T=T, ρ=ρ, z=z, eos=eos.eos, opa=opa.opa, λ_weights=λ_weights, irradiation=Irr, μ_weights=μ_weights, μ_angles=μ_angles,
                    diagonal_inv_operator=lambda_diagonal
				)
			end
		end=#

		# for gradient
		smalldT = 1e-6 .* T
		#update_radiation_z_longchar!(J_plus, F_rad_plus, g_rad_plus, T=T .+ smalldT, ρ=ρ, z=z, eos=eos.eos, opa=opa, λ_weights=λ_weights, irradiation=Irr, μ_weights=μ_weights, μ_angles=μ_angles)
		#update_radiation_z_longchar!(J_minus, F_rad_minus, g_rad_minus, T=T .- smalldT, ρ=ρ, z=z, eos=eos.eos, opa=opa, λ_weights=λ_weights, irradiation=Irr, μ_weights=μ_weights, μ_angles=μ_angles)
		#dFrad_dT .= (F_rad_plus .- F_rad_minus) ./ (2 * smalldT)
		#@show minimum(dFrad_dT) maximum(dFrad_dT) minimum(F_rad) maximum(F_rad)
		#@show minimum(dFrad_dT ./ (4.0 * σ_SB * T.^3)) maximum(dFrad_dT ./ (4.0 * σ_SB * T.^3))
		#dFrad_dT[1:end-1] .= diff(F_rad) ./ diff(T)
		#dFrad_dT[end] = dFrad_dT[end-1]
		dFrad_dT .= 4.0 * σ_SB * T.^3

		#update_mixing_length!(F_conv_plus, v_conv, g_turb, dFconv_dT, T .+ smalldT, P, ρ, τ, eos, exp10(logg); alpha_mlt=α_MLT, Teff=T_eff)
		#update_mixing_length!(F_conv_minus, v_conv, g_turb, dFconv_dT, T .- smalldT, P, ρ, τ, eos, exp10(logg); alpha_mlt=α_MLT, Teff=T_eff)
		update_mixing_length!(F_conv, v_conv, g_turb, dFconv_dT, T, P, ρ, τ, eos, exp10(logg); alpha_mlt=α_MLT, Teff=T_eff)
        #dFconv_dT .= (F_conv_plus .- F_conv_minus) ./ (2 * smalldT)
        #update_temperature_correction_robust!(dT, F_rad, F_conv, dFconv_dT, T, τ, T_eff, lambda_diagonal, J; damping=damping)


		# test new RT
		#@show "convective quantities"
		#@show minimum(F_conv) maximum(F_conv)
		#@show minimum(dFconv_dT) maximum(dFconv_dT)
		#F_conv_m = fill!(similar(F_conv), 0.0)
		#dFconv_dT_m = fill!(similar(F_conv), 0.0)
		chi, chi_ref, S, dSdT = compute_opacities(eos, opa, T, ρ)
		atm = FeutrierRT.Atmosphere(
			T_eff=T_eff, z=z, 
			tau=τ, rho=ρ, Temp=T, 
			#F_conv=F_conv_m, dFconv_dT=dFconv_dT_m,
			F_conv=F_conv, dFconv_dT=dFconv_dT,
			mu=μ_angles, w_mu=μ_weights, w_lambda=λ_weights, 
			chi=chi, chi_ref=chi_ref, B=S, dBdT=dSdT
		)
		#@show minimum(atm.dFconv) maximum(atm.dFconv)
		#FeutrierRT.solve!(atm)
		#FeutrierRT.compute_dT!(atm)
		#@show "solve!"
		#@show minimum(atm.F_bol) maximum(atm.F_bol) 
		#@show minimum(atm.J_bol) maximum(atm.J_bol) 
		#@show minimum(atm.g_rad) maximum(atm.g_rad) 
		#@show minimum(atm.dT) maximum(atm.dT) 

		FeutrierRT.solve_gustafsson!(atm, include_dT=true)
		#@show "solve gustafsson"
		#@show minimum(atm.F_bol) maximum(atm.F_bol) 
		#@show minimum(atm.J_bol) maximum(atm.J_bol) 
		#@show minimum(atm.g_rad) maximum(atm.g_rad) 
		#@show minimum(atm.dT) maximum(atm.dT) 
		J .= atm.J_bol
		F_rad .= atm.F_bol
		g_rad .= atm.g_rad

		#m = (atm.dT .* dT) .< 0.0
		#dT .= atm.dT
		#dT[m] .*= 0.75

		dT .= clamp.(atm.dT, -0.1.*T, 0.1.*T)

		
		


		#update_temperature_correction_mafags!(dT, F_rad, dFrad_dT, F_conv, dFconv_dT, T, T_eff; max_step_frac=0.05, min_deriv=1e-12)
		#update_temperature_correction_robust!(dT, F_rad, F_conv, dFconv_dT, T, τ, T_eff, lambda_diagonal, J; damping=damping)
		#update_temperature_correction_atlas!(dT, F_rad, F_conv, dFconv_dT, T, τ, T_eff; damping=damping)
		#update_temperature_correction!(
		#	dT, T, Q, dQdT, F_rad, F_conv, T_eff
		#)
		#update_temperature_correction_atlas12!(dT, T, τ, T_eff, F_rad, F_conv)
		#compute_auer_temperature_correction!(dT; T=T, ρ=ρ, z=z, eos=eos, opa=opa)

		#compute_temperature_corrections_auer_mihalas!(dT, T, ρ, z, eos, opa, T_eff; μ_nodes=μ_angles, μ_weights=μ_weights, λ_weights=λ_weights, J_comp=J)

		converged = evaluate_iteration!(
			r, iter, maxiter, F_target, dT, τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, T_eff, logg, eos; 
			dFconv_dT=dFconv_dT, J=J,
			J_atm=atm.J_bol, F_atm=atm.F_bol, g_atm=atm.g_rad, dT_atm=atm.dT, S_atm=atm.B[1,:], dSdT_atm=atm.dBdT[1,:], 
			eta1_atm=atm.eta[1,:], eta2_atm=atm.eta[2,:], eta3_atm=atm.eta[3,:], eta4_atm=atm.eta[4,:], 
			eta5_atm=atm.eta[5,:], eta6_atm=atm.eta[6,:], eta7_atm=atm.eta[7,:], eta8_atm=atm.eta[8,:], 
			Q_heat_atm=atm.Q_heat, Q_cool_atm=atm.Q_cool,
			kwargs...
		)
		if converged 
			@info "Atmosphere converged."
			break
		end

		T .+= dT
		T = clamp.(T, 100, 1e12)
		# correct for surface flux
		#F_surface = F_rad[1] # Flux at top node
		#ratio = T_eff / (F_surface / σ_SB) ^0.25 #abs.(F_rad .+ F_conv)
		#@show ratio
		#T .*= ratio

		#force_adiabatic_bottom!(T, P, eos, n_force=20)
		
		# (Keep your existing T smoothing and adiabatic checks here...)
		#=for k in 2:length(T)
			if T[k] < T[k-1]
				T[k] = T[k-1] + 1e-4 
			end
		end=#
		
		update_hydrostatic!(P, ρ, z, T, g_rad, g_turb, τ, eos=eos, logg=logg)
		
	end

    length(r) == 1 ? r[1] : r
end



