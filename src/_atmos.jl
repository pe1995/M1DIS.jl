m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos) = begin
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

function evaluate_iteration!(result, iter, maxiter, F_target, dT, τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, teff, logg, eos; dt_tolerance_rel=0.001, flux_tolerance_rel=0.001, save_every=-1)
	store = save_every > 0 ? ((iter%save_every == 0) | (iter == maxiter)) : false
    F_total = F_rad .+ F_conv
	flux_err_max = maximum(abs.(F_total[2:end-1] .- F_target)) / F_target
	dt_err_max = maximum(abs.(dT[2:end-1] ./ T[2:end-1]))
	sinf = TSO.@sprintf("%4d | %16.4f | %14.4f | %10.1f K\n", 
			iter, flux_err_max*100, dt_err_max*100, maximum(abs.(dT[2:end-1])))
	@info sinf

	converged = (dt_err_max<dt_tolerance_rel) | (flux_err_max<flux_tolerance_rel)
	if converged | store
		append!(result, [m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos)])
	end

	converged
end

"""
    atmosphere(; T_eff, logg, eospath, τ=10 .^range(-5.0, 4, length=100), α_MLT=1.5, maxiter=200)	

Compute a M1DIS atmosphere iteratively based on the given binned opacity table, effective temperature and surface gravity.
"""
function atmosphere(; T_eff, logg, eos, opacity, τ=10 .^range(-5.0, 4, length=100), α_MLT=1.5, maxiter=500, damping=0.4, λ_weights=nothing, T_irradiation=nothing, R_irradiation=nothing, d_irradiation=nothing, use_threads=false, mafags_mlt=false, kwargs...)	
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
		opacity, λ_weights
	end

	T, ρ, P, z = initial_atmosphere(τ, T_eff=T_eff, logg=logg, eos=eos)
	J, F_rad, g_rad = similar(T), similar(T), similar(T)
	F_conv, v_conv, g_turb = similar(T), similar(T), similar(T)
    dFconv_dT = similar(T) 
	dT = similar(T)

	#σ_SB = 5.670374e-5
    F_target = σ_SB * T_eff^4
	#nf = count(log10.(τ) .> 3.)

	# check for irradiation and compute it
	Irr = isnothing(T_irradiation) ? nothing : irradiate(eos, opa, T_irradiation, R_irradiation, d_irradiation)

    @info "============================== M1DIS ===================================="
	@info "iteration | relative flux error (max) | relative T error (max) | ΔT (max)" 
	
	r = []
	for iter in 1:maxiter
		#=if use_threads
			update_radiation_z_longchar_dagger!(
				J, F_rad, g_rad, T=T, ρ=ρ, z=z, eos=eos.eos, opa=opa, λ_weights=λ_weights, irradiation=Irr
			)
		else=#
			update_radiation_z_longchar!(
				J, F_rad, g_rad, T=T, ρ=ρ, z=z, eos=eos.eos, opa=opa, λ_weights=λ_weights
			)
		#end

		if !mafags_mlt
			update_mixing_length!(
				F_conv, v_conv, g_turb, dFconv_dT,
				T, P, ρ, τ, eos, exp10(logg), alpha_mlt=α_MLT
			)
		else
			update_mixing_length_mafags!(
				F_conv, v_conv, g_turb, dFconv_dT,
				T, P, ρ, τ, eos, exp10(logg), alpha_mlt=α_MLT, Teff=T_eff
			)
		end

		update_temperature_correction_atlas!(dT, F_rad, F_conv, T, τ, T_eff, damping=damping)

		converged = evaluate_iteration!(r, iter, maxiter, F_target, dT, τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, T_eff, logg, eos; kwargs...)
		if converged 
			@info "Atmosphere converged."
			break
		end

		T .+= dT
		#force_adiabatic_bottom!(T, P, eos, n_force=nf)
		
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



