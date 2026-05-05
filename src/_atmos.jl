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
    damping=0.01, 
    v_mac=0.0,
    pbeta=1.0,
    T_irradiation=nothing, R_irradiation=nothing, d_irradiation=nothing, F_irradiation=nothing,
    T=nothing, ρ=nothing, P=nothing, z=nothing, 
    feutrier=true,
    use_threads=false,
    scattering_opacity=nothing,
    target_flux=nothing,
    steepness=15.0,
    tau_trans=-2.0,
    solver=:vef,
    stabilize_convection=false,
    kwargs...)  

    @optionalTiming initialization_time begin
        # --- EoS & Opacity Setup ---
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
            @verbose_warn 2 "Opacity was passed without wrapping it into an ExtendedOpacity. Guessing: $(o.binned ? "binned" : "not binned")"
            o
        else
            opacity
        end
        
        @assert typeof(opacity) <: Union{TSO.ExtendedOpacity, TSO.MiniOpacityTable} "Opacity must be an ExtendedOpacity or MiniOpacityTable."
        if opacity.binned
            @assert typeof(opacity) <: TSO.ExtendedOpacity "Binned opacities must use ExtendedOpacity."
            if !haskey(opacity.extensions, :dS_dT)
                TSO.gradients!(TSO.table(eos), opacity)
            end
        else
            use_threads = true
        end

        if !isnothing(scattering_opacity)
            @assert typeof(scattering_opacity) <: TSO.MiniOpacityTable "Scattering opacity must be MiniOpacityTable."
            @info "Running with scattering treatment."
        end

        # --- Initial State ---
        τ = deepcopy(τ)
        T, ρ, P, z = if isnothing(T) 
            initial_atmosphere(τ, T_eff=T_eff, logg=logg, eos=eos)
        else
            deepcopy(T), deepcopy(ρ), deepcopy(P), deepcopy(z)
        end
        
        F_target = isnothing(target_flux) ? σ_SB * T_eff^4 : target_flux
		teff_target = isnothing(target_flux) ? T_eff : (target_flux / σ_SB) ^ 0.25
        μ_angles, μ_weights = generate_mu_grid(4)
        Irr = isnothing(d_irradiation) ? nothing : irradiate(eos, opacity, T_irradiation, R_irradiation, d_irradiation, F_irradiation)

        # --- Pre-compute initial opacities to bootstrap the Atmosphere object ---
		@optionalTiming prepare_opacities_time begin
			chi, chi_ref, S, dSdT, dchidT, chi_scat = if feutrier
				c, c_ref, s_val, dsdt_val, dchidt_val = opacity.binned ? compute_opacities(eos, opacity, T, ρ) : compute_opacities_chunked(eos, opacity, T, ρ)
				c_scat = (!isnothing(scattering_opacity)) ? compute_opacities_chunked(eos, scattering_opacity, T, ρ, opacity_only=true)[1] : nothing
				c, c_ref, s_val, dsdt_val, dchidt_val, c_scat
			else
				nothing, nothing, nothing, nothing, nothing, nothing
			end
		end

		# --- Master Storage Initialization ---
        atm = Atmosphere(
            T_eff=teff_target, z=z, tau=τ, rho=ρ, Temp=T, P_gas=P,
            mu=μ_angles, w_mu=μ_weights, 
            chi=chi, chi_ref=chi_ref, B=S, dBdT=dSdT, dchidT=dchidT, I_top=Irr, chi_scat=chi_scat
        )
        
        @verbose_info 2 "================================= M1DIS ================================="
        if typeof(opacity) <: TSO.MiniOpacityTable
            @verbose_info 2 "Running M1DIS with MiniOpacityTable. Source function is computed on the fly."
        end
        if !opacity.binned
            @verbose_info 2 "Running M1DIS with unbinned opacity table. Forcing use_threads=true."
		end

        # check selected solver
        solver = if solver in [:gustafsson, :vef, :vef_full, :approximate]
            solver
        else
            @verbose_info 1 "Selected solver $(solver) not available. Switching to default solver."
            :vef
        end
        @verbose_info 1 "Running RT solver:$(solver) with $(Base.Threads.nthreads()) threads."
        
        @verbose_info 2 "========================================================================="
        sinf = TSO.@sprintf(
            "%s | %s | %s | %s | %s | %s\n________________________________________________________________________________________________________\n", 
            "iteration", "ΔF/F (max, %)", "log(τ) of max. ΔF/F", "ΔT/T (max, %)", "log(τ) of max. ΔT/T", "ΔT (max, K)"
        )
        @verbose_info 1 sinf
    end
	
	flux_err_max_prev = Inf
	flux_err_max_curr = Inf
    stabilizer_stage = 3

    # MARCS-standard thresholds
    tcmxu_inv = 1.0 / damping 
    tcmxu_top_inv = 1.0 / (damping * 5.0) 
    r = []

    @optionalTiming relaxation_time for iter in 1:maxiter
        try
            # MLT
            @optionalTiming mixing_length_time begin
                update_mixing_length!(
                    atm.F_conv, 
                    atm.v_conv, 
                    atm.P_rad, 
                    atm.P_turb, 
                    atm.dFconv_dT, 
                    atm.Temp, 
                    atm.P_gas, 
                    atm.rho, 
                    atm.tau, 
                    eos, 
                    exp10(logg); 
                    alpha_mlt=α_MLT, Teff=teff_target, v_mac=v_mac*1e5, pbeta=pbeta
                )
            end

            if stabilize_convection
                # Stabilizer for the approximate solver
                if solver==:approximate
                    stabilizer_stage = if (flux_err_max_prev > 50.0)
                        stabilizer_stage > 3 ? 3 : stabilizer_stage
                    elseif (flux_err_max_prev > 1.0)
                        stabilizer_stage > 2 ? 2 : stabilizer_stage
                    else
                        stabilizer_stage > 1 ? 1 : stabilizer_stage
                    end

                    if stabilizer_stage == 3
                        for n in 2:length(atm.F_conv)
                            if ((atm.F_conv[n-1] > 0.0) && (atm.F_conv[n] < atm.F_conv[n-1]))
                                atm.F_conv[n] = atm.F_conv[n-1]
                                atm.v_conv[n] = atm.v_conv[n-1]
                                atm.P_turb[n] = atm.P_turb[n-1]
                                atm.dFconv_dT[n] = atm.dFconv_dT[n-1]
                            end
                        end
                        smooth_array!(atm.F_conv, passes=1)
                        smooth_array!(atm.v_conv, passes=1)
                        smooth_array!(atm.P_turb, passes=1)
                        smooth_array!(atm.dFconv_dT, passes=1)
                    elseif stabilizer_stage == 2
                        for n in 2:length(atm.F_conv)
                            if ((atm.dFconv_dT[n-1] > 0.0) && (atm.dFconv_dT[n] < atm.dFconv_dT[n-1]))
                                atm.dFconv_dT[n] = atm.dFconv_dT[n-1]
                                atm.P_turb[n] = atm.P_turb[n-1]
                            end
                        end
                        smooth_array!(atm.dFconv_dT, passes=1)
                        smooth_array!(atm.P_turb, passes=1)
                    end
                else
                    stabilizer_stage = if (flux_err_max_prev > 50.0)
                        stabilizer_stage > 3 ? 3 : stabilizer_stage
                    elseif (flux_err_max_prev > 1.0)
                        stabilizer_stage > 2 ? 2 : stabilizer_stage
                    else
                        stabilizer_stage > 1 ? 1 : stabilizer_stage
                    end

                    if stabilizer_stage == 3
                        smooth_array!(atm.F_conv, passes=1)
                        smooth_array!(atm.v_conv, passes=1)
                        smooth_array!(atm.P_turb, passes=1)
                        smooth_array!(atm.dFconv_dT, passes=1)
                    elseif stabilizer_stage == 2
                        smooth_array!(atm.dFconv_dT, passes=1)
                        smooth_array!(atm.P_turb, passes=1)
                    end
                end
            end
            
            # Radiative Transfer
            @optionalTiming radiation_transfer_time begin
                @optionalTiming compute_opacities_time if opacity.binned
                    compute_opacities!(atm.chi, atm.chi_ref, atm.B, atm.dBdT, atm.dchidT, eos, opacity, atm.Temp, atm.rho)
                else
                    compute_opacities_chunked!(atm.chi, atm.chi_ref, atm.B, atm.dBdT, atm.dchidT, eos, opacity, atm.Temp, atm.rho)
                end

                if !isnothing(scattering_opacity)
                    @optionalTiming compute_opacities_time compute_opacities_chunked!(atm.chi_scat, nothing, nothing, nothing, nothing, eos, scattering_opacity, atm.Temp, atm.rho)
                end

                @optionalTiming update_atmosphere_time update!(atm)
                @optionalTiming solve_RT_time if solver == :gustafsson
                    solve_gustafsson!(atm)
                elseif solver == :vef
                    solve_VEF!(atm)
                elseif solver == :vef_full
                    solve_VEF_full!(atm)
                else
                    solve_approximate!(atm; steepness=steepness, tau_trans=tau_trans)
                end

                # Update flux errors inside atm (Staggered Flux sum)
                atm.F_total .= atm.F_rad .+ atm.F_conv
                atm.F_err_rel .= (atm.F_total .- F_target) ./ F_target
                flux_err_max_curr = maximum(abs.(atm.F_err_rel))
            end

            # Damping
            for i in 1:length(atm.dT)
                #scale = flux_err_max_curr > flux_err_max_prev ? tcmxu_inv * 2 : tcmxu_inv
                scale = tcmxu_inv
                atm.dT[i] = atm.dT[i] / sqrt(1.0 + (scale * atm.dT[i] / atm.Temp[i])^2)
                #atm.dT[i] = clamp(atm.dT[i], -damping * atm.Temp[i], damping * atm.Temp[i])
            end
            flux_err_max_prev = flux_err_max_curr

            # Evaluation
            converged = evaluate_iteration!(
                r, iter, maxiter, F_target, 
                atm.dT, 
                atm.tau, 
                atm.z, 
                atm.Temp, 
                atm.rho, 
                atm.P_gas, 
                atm.F_rad, 
                atm.F_conv, 
                atm.dFconv_dT, 
                T_eff, 
                logg, 
                eos, 
                damping; 
                J=atm.J_bol, 
                g_rad=atm.g_rad, 
                P_turb=atm.P_turb, 
                P_rad=atm.P_rad, 
                F_err_rel=atm.F_err_rel, 
                Q_rad=atm.Q_rad,
                chi_max=maximum(atm.chi, dims=1), chi_min=minimum(atm.chi, dims=1),
                kwargs...
            )

            if converged 
                @verbose_info 1 "Atmosphere converged."
                break
            end

            # Apply corrections & Hydrostatic Equilibrium
            atm.Temp .+= atm.dT
            atm.Temp .= clamp.(atm.Temp, 10, 1e12)
            
            @optionalTiming hydrostatic_time update_hydrostatic!(
                atm.P_gas, 
                atm.rho, 
                atm.z, 
                atm.Temp, 
                atm.P_turb, 
                atm.P_rad, 
                atm.tau, 
                eos=eos, 
                logg=logg
            )
        catch e
            @warn "M1DIS failed. Error: $e"
            break
        end
    end

    return length(r) == 1 ? r[1] : r
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
	P_rad = similar(T_initial)
	P_turb = similar(T_initial)

    z_initial .= 0.0
    ρ_initial .= 0.0
	P_initial .= 0.0
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
	#dt_tolerance_rel=0.00001, dt_tolerance=0.001, flux_tolerance_rel=0.01, save_every=1, kwargs...)
	dt_tolerance_rel=0.00001, dt_tolerance=1.0, flux_tolerance_rel=0.01, save_every=1, kwargs...)
	
    # store the atmosphere every `save_every` iterations
	store = save_every > 0 ? ((iter%save_every == 0) | (iter == maxiter)) : false
    F_total = F_rad .+ F_conv
	flux_err_max = maximum(abs.(F_total .- F_target) / F_target)
    f_amax = argmax(abs.(F_total .- F_target))
	dt_err_max = maximum(abs.(dT ./ T))
	dt_amax = argmax(abs.(dT ./ T))
	dt_err_max_abs = maximum(abs.(dT))
	
    sinf = TSO.@sprintf(
        "%4d | %20.4f %% | %6.2f | %14.4f %% | %6.2f | %14.4f K\n", 
		iter, flux_err_max*100, log10(τ[f_amax]), dt_err_max*100, log10(τ[dt_amax]), dt_err_max_abs
    )
	@verbose_info 1 sinf

	converged = (dt_err_max<dt_tolerance_rel) | (flux_err_max<flux_tolerance_rel)
	#converged = (dt_err_max<dt_tolerance) && (flux_err_max<flux_tolerance_rel)
	if converged | store
		append!(result, [m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos; kwargs...)])
	end

	converged
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

function save!(model_data::MUST.Box, model_name; eos500=nothing, folder="./", vmic=0.0, logg=4.5, information=nothing)
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
		MUST.save_text_m1d(f_new_m1d, tau500, T, Ne; logg=logg, header=model_name, vmic=vmic_arr, information=information)
		
		f_new_dscale = joinpath(run_i, "dscale.$(model_name)")
		MUST.save_text_m1d_dscale(f_new_dscale, tau500; header=model_name)
	end

	# 3. HDF5 format
	MUST.flip!(model_data)
	MUST.save(model_data, folder=run_i, name=model_name)

    return run_i
end



