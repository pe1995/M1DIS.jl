# ============================================================================
# Iterative computation of the Atmosphere
# ============================================================================

"""
    atmosphere(; T_eff, logg, eos, opacity, kwargs...)

Compute a 1D model atmosphere iteratively using the given opacity table,
effective temperature, and surface gravity.

The function initialises a gray atmosphere, then repeatedly:
  1. Solves MLT convection
  2. Updates opacities
  3. Solves the radiative transfer equation
  4. Applies a damped temperature correction
  5. Integrates hydrostatic equilibrium

and terminates when either the flux or temperature correction falls below
the specified tolerance.

# Required arguments
- `T_eff`    — effective temperature [K]
- `logg`     — log₁₀ of surface gravity [cgs]
- `eos`      — equation of state (`TSO.ExtendedEoS` or compatible)
- `opacity`  — opacity table (`TSO.ExtendedOpacity` or `TSO.MiniOpacityTable`)

# Iteration control
- `τ`                  — optical depth grid (default: 10^(-6..2), 100 points)
- `maxiter`            — maximum number of iterations (default: 20)
- `damping`            — MARCS-style damping factor for ΔT (default: 0.01)
- `dt_tolerance`       — absolute ΔT convergence threshold [K] (default: 0.5)
- `flux_tolerance_rel` — relative flux convergence threshold (default: 0.01)
- `save_every`         — store a snapshot every N iterations (default: 1)

# Physics options
- `α_MLT`              — mixing-length parameter (default: 1.5)
- `v_mac`              — microturbulent velocity for turbulent pressure [km/s] (default: 0.0)
- `pbeta`              — turbulent pressure coefficient (default: 1.0)
- `convection`         — enable convection (default: true)
- `stabilize_convection` — apply convection smoothing heuristic (default: false)
- `solver`             — RT solver symbol: `:vef`, `:vef_full`, `:vef_mod`,
                         `:gustafsson`, `:approximate` (default: `:vef`)
- `vef_mode`           — dT mode for VEF: `:boundary`, `:RE`, `:FC`, `:switch`
                         (default: `:boundary`)
- `tau_trans`          — log₁₀(τ) transition for `:switch` mode (default: -2.0)
- `steepness`          — steepness for approximate solver (default: 15.0)
- `scattering_opacity` — additional scattering opacity (`TSO.MiniOpacityTable`)

# Starting atmosphere (all optional; gray atmosphere used if omitted)
- `T`, `ρ`, `P`, `z` — initial temperature, density, pressure, height profiles

# Irradiation (all optional)
- `T_irradiation`, `R_irradiation`, `d_irradiation`, `F_irradiation`

# Returns
- A single `MUST.Box` if only one snapshot is stored, otherwise a `Vector{MUST.Box}`.
"""
function atmosphere(;
        T_eff, logg, eos, opacity,
        τ = 10 .^ range(-6.0, 2.0, length=100),
        α_MLT    = 1.5,
        maxiter  = 20,
        damping  = 0.01,
        v_mac    = 0.0,
        pbeta    = 1.0,
        T_irradiation = nothing, R_irradiation = nothing,
        d_irradiation = nothing, F_irradiation = nothing,
        T = nothing, ρ = nothing, P = nothing, z = nothing,
        feautrier           = true,
        use_threads         = false,
        scattering_opacity  = nothing,
        target_flux         = nothing,
        steepness           = 15.0,
        tau_trans           = -2.0,
        solver              = :vef,
        vef_mode            = :boundary,
        stabilize_convection = false,
        convection          = true,
        kwargs...)

    # Initialisation
    @optionalTiming initialization_time begin
        eos, opacity, use_threads = _prepare_eos_opacity(eos, opacity, scattering_opacity, use_threads)

        τ, T, ρ, P, z = _setup_initial_state(τ, T_eff, logg, eos, T, ρ, P, z)

        F_target    = isnothing(target_flux) ? σ_SB * T_eff^4 : target_flux
        teff_target = isnothing(target_flux) ? T_eff : (target_flux / σ_SB)^0.25

        Irr = isnothing(d_irradiation) ? nothing : irradiate(eos, opacity, T_irradiation, R_irradiation, d_irradiation, F_irradiation)

        @optionalTiming prepare_opacities_time begin
            chi, chi_ref, S, dSdT, dchidT, chi_scat, dchidT_scat = _compute_initial_opacities(feautrier, eos, opacity, scattering_opacity, T, ρ)
        end

        Nf = size(chi, 1)

        # Construct the atmosphere (all physics arrays zero-initialized, angles default to 4-pt GL)
        atm = Atmosphere(τ, Nf; T_eff=teff_target, with_scattering=!isnothing(chi_scat))

        # Populate physical state from the initial atmosphere
        populate!(atm;
            Temp=T, rho=ρ, z=z, P_gas=P,
            chi=chi, chi_ref=chi_ref,
            B=S, dBdT=dSdT, dchidT=dchidT,
        )
        isnothing(chi_scat) || populate!(atm; chi_scat=chi_scat, dchidT_scat=dchidT_scat)
        isnothing(Irr)      || populate!(atm; I_top=Irr)

        # Scratch arrays for scattering opacity updates (reused each iteration)
        scattering_bin_z   = similar(chi_ref)
        scattering_bin_lam = isnothing(chi_scat) ? nothing : similar(chi_scat)

        # Derive tau_lambda from z and chi, then print header
        update!(atm)

        solver = _validate_solver(solver)
        _print_run_header(atm, opacity, solver, vef_mode)
    end

    # Iteration
    results             = []
    flux_err_max_prev   = Inf
    flux_err_max_curr   = Inf
    flux_err_prev       = similar(atm.F_total)
    dt_limiter          = similar(atm.F_total)
    stabilizer_stage    = 3
    tcmxu_inv           = 1.0 / damping

    flux_err_prev .= Inf
    dt_limiter .= 1.0

    current_tau_trans = vef_mode == :switch ? log10(atm.tau[2]) : tau_trans


    @optionalTiming relaxation_time for iter in 1:maxiter
        use_convection = convection #&& (iter > 5)

        try
            # Convection
            if use_convection
                @optionalTiming mixing_length_time begin
                    update_mixing_length!(
                        atm.F_conv, atm.v_conv, atm.P_rad, atm.P_turb, atm.dFconv_dT,
                        atm.Temp, atm.P_gas, atm.rho, atm.tau,
                        eos, exp10(logg);
                        alpha_mlt=α_MLT, Teff=teff_target, v_mac=v_mac*1e5, pbeta=pbeta
                    )
                end

                if stabilize_convection
                    _stabilize_convection!(atm, flux_err_max_prev, stabilizer_stage, solver)
                end
            else
                populate!(atm, F_conv=0.0, dFconv_dT=0.0, v_conv=0.0, P_turb=0.0)
            end

            # Radiative Transfer
            @optionalTiming radiation_transfer_time begin
                @optionalTiming compute_opacities_time _update_opacities!(
                    atm, eos, opacity, scattering_opacity,
                    scattering_bin_z, scattering_bin_lam
                )

                @optionalTiming update_atmosphere_time update!(atm)

                @optionalTiming solve_RT_time _run_solver!(atm, solver, vef_mode, current_tau_trans, steepness)

                atm.F_total   .= atm.F_rad .+ atm.F_conv
                atm.F_err_rel .= (atm.F_total .- F_target) ./ F_target
                flux_err_max_curr = maximum(abs.(atm.F_err_rel))
            end

            if vef_mode == :switch && iter > 1
                improvement = (flux_err_max_prev - flux_err_max_curr) / max(flux_err_max_prev, 1e-10)
                if improvement < 0.05
                    current_tau_trans = min(-0.5, current_tau_trans + 0.2)
                end
            end

            # Damping
            _apply_dT_damping!(atm, tcmxu_inv, damping, flux_err_max_curr, flux_err_max_prev)

            # Convergence evaluation 
            converged = evaluate_iteration!(
                results, atm, iter, maxiter, F_target, T_eff, logg, eos, vef_mode;
                J=atm.J_bol, 
                g_rad=atm.g_rad, 
                P_turb=atm.P_turb, 
                P_rad=atm.P_rad, 
                F_err_rel=atm.F_err_rel, 
                Q_rad=atm.Q_rad,
                chi_max=maximum(atm.chi, dims=1), 
                chi_min=minimum(atm.chi, dims=1),
                kwargs...
            )

            if converged
                header_line = "──────────┼──────────────────────┼─────────────────────┼──────────────────────┼─────────────────────┼─────────────────"
                print_nice(header_line, category="Atmosphere", color=color_messages[], verbosity=1)
                print_nice("✅ Atmosphere converged with $(round(flux_err_max_curr * 100, digits=2))% flux error.", category="Atmosphere", color=color_messages[], verbosity=1)
                break
            end

            # Temperature correction & hydrostatic equilibrium
            #for i in eachindex(atm.dT)
            #    if atm.F_err_rel[i] > flux_err_prev[i]
            #        dt_limiter[i] = 0.2
            #   else
            #        dt_limiter[i] = 1.0
            #    end

            atm.Temp .+= atm.dT
            atm.Temp  .= clamp.(atm.Temp, 10, 1e12)
            #end
        
            flux_err_max_prev = flux_err_max_curr
            flux_err_prev .= atm.F_err_rel

            @optionalTiming hydrostatic_time update_hydrostatic!(
                atm.P_gas, atm.rho, atm.z, atm.Temp, atm.P_turb, atm.P_rad, atm.tau,
                eos=eos, logg=logg
            )

        catch e
            @warn "M1DIS iteration $iter failed: $e"
            break
        end
    end

    return length(results) == 1 ? results[1] : results
end

# ============================================================================
# Private helpers for atmosphere()
# ============================================================================

function _prepare_eos_opacity(eos, opacity, scattering_opacity, use_threads)
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
        print_nice("Running with scattering treatment.", category="Atmosphere", color=color_messages[], verbosity=2)
    end

    return eos, opacity, use_threads
end

function _setup_initial_state(τ, T_eff, logg, eos, T, ρ, P, z)
    τ = deepcopy(τ)
    if isnothing(T)
        T, ρ, P, z = initial_atmosphere(τ, T_eff=T_eff, logg=logg, eos=eos)
    else
        T, ρ, P, z = deepcopy(T), deepcopy(ρ), deepcopy(P), deepcopy(z)
    end
    return τ, T, ρ, P, z
end

function _compute_initial_opacities(feautrier, eos, opacity, scattering_opacity, T, ρ)
    if !feautrier
        return nothing, nothing, nothing, nothing, nothing, nothing, nothing
    end

    chi, chi_ref, S, dSdT, dchidT =
        opacity.binned ? compute_opacities(eos, opacity, T, ρ) :
                         compute_opacities_chunked(eos, opacity, T, ρ)

    chi_scat = isnothing(scattering_opacity) ? nothing :
        compute_opacities_chunked(eos, scattering_opacity, T, ρ, opacity_only=true)[1]

    dchidT_scat = isnothing(chi_scat) ? nothing : similar(chi_scat)

    return chi, chi_ref, S, dSdT, dchidT, chi_scat, dchidT_scat
end

function _validate_solver(solver)
    valid = (:gustafsson, :vef, :vef_full, :vef_mod, :approximate)
    if solver ∉ valid
        print_nice(
            "Selected solver $(solver) not available. Switching to default :vef.",
            category="Atmosphere", color=color_messages[], verbosity=1)
        return :vef
    end
    return solver
end

function _print_run_header(atm, opacity, solver, vef_mode)
    if any(>(0.0), atm.I_top)
        print_nice("External irradiation has been turned on.",
            category="Atmosphere", color=color_messages[], verbosity=1)
    end
    header_names = TSO.@sprintf(
        "%-9s │ %-20s │ %-19s │ %-20s │ %-19s │ %-16s",
        "iteration", "ΔF/F (max, %)", "log(τ) of max. ΔF/F",
        "ΔT/T (max, %)", "log(τ) of max. ΔT/T", "ΔT (max, K)"
    )
    header_line = "──────────┼──────────────────────┼─────────────────────┼──────────────────────┼─────────────────────┼─────────────────"
    print_nice(header_names, category="Atmosphere", color=color_messages[], verbosity=1)
    print_nice(header_line, category="Atmosphere", color=color_messages[], verbosity=1)
end

function _update_opacities!(atm, eos, opacity, scattering_opacity, scratch_z, scratch_lam)
    if opacity.binned
        compute_opacities!(atm.chi, atm.chi_ref, atm.B, atm.dBdT, atm.dchidT,
                           eos, opacity, atm.Temp, atm.rho)
    else
        compute_opacities_chunked!(atm.chi, atm.chi_ref, atm.B, atm.dBdT, atm.dchidT,
                                   eos, opacity, atm.Temp, atm.rho)
    end

    if !isnothing(scattering_opacity)
        compute_opacities_chunked!(atm.chi_scat, scratch_z, scratch_lam, scratch_lam, atm.dchidT_scat,
                                   eos, scattering_opacity, atm.Temp, atm.rho)
    end
end

function _run_solver!(atm, solver, vef_mode, tau_trans, steepness)
    if solver == :gustafsson
        solve_gustafsson!(atm)
    elseif solver == :vef
        solve_VEF!(atm, mode=vef_mode, tau_trans=tau_trans)
    elseif solver == :vef_full
        solve_VEF_full!(atm, mode=vef_mode, tau_trans=tau_trans)
    elseif solver == :vef_mod
        solve_VEF_mod!(atm)
    else  # :approximate
        solve_approximate!(atm; steepness=steepness, tau_trans=tau_trans)
    end
end

function _apply_dT_damping!(atm, tcmxu_inv, damping, flux_err_max_curr, flux_err_max_prev)
   tcmxu_inv_loc = if flux_err_max_curr > flux_err_max_prev
        tcmxu_inv * 5.0
    else
        tcmxu_inv
    end

    for i in eachindex(atm.dT)
        atm.dT[i] = atm.dT[i] / sqrt(1.0 + (tcmxu_inv_loc * atm.dT[i] / atm.Temp[i])^2)
    end
    #atm.dT .= clamp.(atm.dT, -damping*atm.Temp, damping*atm.Temp)
end

function _stabilize_convection!(atm, flux_err_max_prev, stabilizer_stage, solver)
    stabilizer_stage = if flux_err_max_prev > 50.0
        stabilizer_stage > 3 ? 3 : stabilizer_stage
    elseif flux_err_max_prev > 1.0
        stabilizer_stage > 2 ? 2 : stabilizer_stage
    else
        stabilizer_stage > 1 ? 1 : stabilizer_stage
    end

    if solver == :approximate
        if stabilizer_stage == 3
            for n in 2:length(atm.F_conv)
                if (atm.F_conv[n-1] > 0.0) && (atm.F_conv[n] < atm.F_conv[n-1])
                    atm.F_conv[n]    = atm.F_conv[n-1]
                    atm.v_conv[n]    = atm.v_conv[n-1]
                    atm.P_turb[n]    = atm.P_turb[n-1]
                    atm.dFconv_dT[n] = atm.dFconv_dT[n-1]
                end
            end
            smooth_array!(atm.F_conv,    passes=1)
            smooth_array!(atm.v_conv,    passes=1)
            smooth_array!(atm.P_turb,    passes=1)
            smooth_array!(atm.dFconv_dT, passes=1)
        elseif stabilizer_stage == 2
            for n in 2:length(atm.F_conv)
                if (atm.dFconv_dT[n-1] > 0.0) && (atm.dFconv_dT[n] < atm.dFconv_dT[n-1])
                    atm.dFconv_dT[n] = atm.dFconv_dT[n-1]
                    atm.P_turb[n]    = atm.P_turb[n-1]
                end
            end
            smooth_array!(atm.dFconv_dT, passes=1)
            smooth_array!(atm.P_turb,    passes=1)
        end
    else
        if stabilizer_stage == 3
            smooth_array!(atm.F_conv,    passes=1)
            smooth_array!(atm.v_conv,    passes=1)
            smooth_array!(atm.P_turb,    passes=1)
            smooth_array!(atm.dFconv_dT, passes=1)
        elseif stabilizer_stage == 2
            smooth_array!(atm.dFconv_dT, passes=1)
            smooth_array!(atm.P_turb,    passes=1)
        end
    end
end

# ============================================================================
# Initial Atmosphere
# ============================================================================

function initial_atmosphere(τ_grid; T_eff, logg, eos)
    # Gray atmosphere (Eddington approximation)
    T_initial = T_eff .* (0.75 .* (τ_grid .+ 2/3)) .^ 0.25

    ρ_initial = similar(T_initial)
    P_initial = similar(T_initial)
    z_initial = similar(T_initial)
    P_rad     = similar(T_initial)
    P_turb    = similar(T_initial)

    z_initial .= 0.0; ρ_initial .= 0.0; P_initial .= 0.0
    P_rad .= 0.0;     P_turb .= 0.0

    update_hydrostatic!(P_initial, ρ_initial, z_initial, T_initial, P_turb, P_rad, τ_grid,
                        logg=logg, eos=eos)

    return T_initial, ρ_initial, P_initial, z_initial
end

# ============================================================================
# Iteration Evaluation
# ============================================================================

function evaluate_iteration!(results,
        atm::Atmosphere, iter::Int, maxiter::Int,
        F_target, T_eff, logg, eos, vef_mode;
        dt_tolerance::Float64 = 0.5,
        dt_tolerance_rel::Float64  = 0.001,
        flux_tolerance_rel::Float64 = 0.001,
        save_every::Int           = 1,
        kwargs...)

    F_total     = atm.F_rad .+ atm.F_conv
    flux_err    = abs.(F_total .- F_target) ./ F_target
    dT_rel      = abs.(atm.dT ./ atm.Temp)

    flux_err_max    = maximum(flux_err)
    flux_err_idx    = argmax(flux_err)
    dT_rel_max      = maximum(dT_rel)
    dT_rel_idx      = argmax(dT_rel)
    dT_abs_max      = maximum(abs.(atm.dT))

    sinf = TSO.@sprintf(
        "%9d │ %18.4e %% │ %19.2f │ %18.4e %% │ %19.2f │ %14.4e K",
        iter,
        flux_err_max * 100, log10(atm.tau[flux_err_idx]),
        dT_rel_max   * 100, log10(atm.tau[dT_rel_idx]),
        dT_abs_max
    )
    print_nice(sinf, category="Atmosphere", color=color_messages[], verbosity=1)

    #converged = (dT_abs_max < dt_tolerance) || (flux_err_max < flux_tolerance_rel)
    converged = flux_err_max < flux_tolerance_rel

    converged = if vef_mode in [:switch, :RE]
        converged && (dT_rel_max < dt_tolerance_rel)
    else
        converged
    end

    store = (save_every > 0) && ((iter % save_every == 0) || (iter == maxiter) || converged)
    if store
        push!(results, _build_result(atm, T_eff, logg, eos; kwargs...))
    end

    return converged
end

function _build_result(atm::Atmosphere, T_eff, logg, eos; kwargs...)
    m1disBox(
        atm.tau, atm.z, atm.Temp, atm.rho, atm.P_gas,
        atm.F_rad, atm.F_conv, atm.dFconv_dT, atm.dT,
        T_eff, logg, eos; kwargs...
    )
end

# ============================================================================
# Return the M1DIS.jl result in the same format as in DISPATCH
# ============================================================================

m1disBox(τ, z, T, ρ, P, F_rad, F_conv, dFconv_dT, dT, teff, logg, eos; kwargs...) = begin
    p  = MUST.AtmosphericParameters(-99.0, Base.convert(Float64, teff), Base.convert(Float64, logg), Dict{Symbol,Float64}())
    zz = reshape(z, 1, 1, :) |> deepcopy
    xx = zeros(size(zz))
    yy = zeros(size(zz))

    τ_new = deepcopy(τ)

    d = Dict(
        :τ_ross    => reshape(τ_new,      1, 1, :) |> deepcopy,
        :T         => reshape(T,          1, 1, :) |> deepcopy,
        :d         => reshape(ρ,          1, 1, :) |> deepcopy,
        :Pg        => reshape(P,          1, 1, :) |> deepcopy,
        :F_rad     => reshape(F_rad,      1, 1, :) |> deepcopy,
        :F_conv    => reshape(F_conv,     1, 1, :) |> deepcopy,
        :dFconv_dT => reshape(dFconv_dT,  1, 1, :) |> deepcopy,
        :dT        => reshape(dT,         1, 1, :) |> deepcopy,
    )

    for (k, v) in kwargs
        if typeof(v) <: AbstractArray
            d[k] = reshape(v, 1, 1, :) |> deepcopy
        else
            @warn "Kwarg $k of type $(typeof(v)) is not an array and will not be saved."
        end
    end

    MUST.Box(xx, yy, zz, d, p)
end

# ============================================================================
# Saving the M1DIS.jl result in Multi3D and Multi1D format
# ============================================================================

function save!(model_data::MUST.Box, model_name; eos500=nothing, folder="./", vmic=0.0, logg=4.5, information=nothing)
    base_path = abspath(folder)
    isdir(base_path) || mkpath(base_path)

    run_i = joinpath(base_path, model_name)
    isdir(run_i) || mkdir(run_i)

    model_data.z .= model_data.z .- TSO.optical_surface(model_data.data[:τ_ross][1,1,:], model_data.z[1,1,:])
    MUST.flip!(model_data, depth=true)

    b       = model_data
    z       = b[:z][1,1,:]
    rho     = b[:d][1,1,:]
    T       = b[:T][1,1,:]
    vmic_arr = fill(vmic, length(z))

    # 1. M3D format
    f_new = joinpath(run_i, "$(model_name)_m3d.txt")
    MUST.save_text_m3d(f_new, z, rho, T; header=model_name, vmic=vmic_arr)

    # 2. M1D format
    if !isnothing(eos500)
        MUST.flip!(model_data, depth=false)
        model_data.data[:Ne]   = reshape(TSO.lookup(eos500, :lnNe,   log.(model_data[:d]), log.(model_data[:T])) .|> exp, 1, 1, :)
        model_data.data[:κ500] = reshape(TSO.lookup(eos500, :lnRoss, log.(model_data[:d]), log.(model_data[:T])) .|> exp, 1, 1, :)
        model_data.data[:τ500] = MUST.optical_depth(model_data, opacity=:κ500, density=:d)
        MUST.flip!(model_data, depth=true)

        tau500  = log10.(b[:τ500][1,1,:])
        Ne      = b[:Ne][1,1,:]

        f_new_m1d    = joinpath(run_i, "atmos.$(model_name)")
        f_new_dscale = joinpath(run_i, "dscale.$(model_name)")

        MUST.save_text_m1d(f_new_m1d,     tau500, T, Ne; logg=logg, header=model_name, vmic=vmic_arr, information=information)
        MUST.save_text_m1d_dscale(f_new_dscale, tau500; header=model_name)
    end

    # 3. HDF5 format
    MUST.flip!(model_data)
    MUST.save(model_data, folder=run_i, name=model_name)

    return run_i
end
