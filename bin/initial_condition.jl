cubic_regression_machines = Dict(
    "bottom_rho" => abspath(joinpath(dirname(pathof(MUST)), "..", "initial_grids", "Stagger", "stagger_bottom_rho_cubic.jlso")),
    "bottom_T" => abspath(joinpath(dirname(pathof(MUST)), "..", "initial_grids", "Stagger", "stagger_bottom_T_cubic.jlso")),
    "top_rho" => abspath(joinpath(dirname(pathof(MUST)), "..", "initial_grids", "Stagger", "stagger_top_rho_cubic.jlso")),
    "top_T" => abspath(joinpath(dirname(pathof(MUST)), "..", "initial_grids", "Stagger", "stagger_top_T_cubic.jlso")),
)

# ============================================================================
# TSO 1D model from box
# ============================================================================

function box2model(b)
    lt, lnrho = MUST.profile(MUST.mean, b, :log10τ_ross, :logd)
    _, lnT = MUST.profile(MUST.mean, b, :log10τ_ross, :logT)
    _, z = MUST.profile(MUST.mean, b, :log10τ_ross, :z)
    TSO.flip(
        TSO.Model1D(z=z, τ=exp10.(lt), lnρ=lnrho, lnT=lnT, logg=Float32(b.parameter.logg)), 
        depth=true
    )
end

# ============================================================================
# ML regression to predict boundary conditions from stagger
# ============================================================================

function predict_from_machine(teff, logg, feh; offsets=nothing)
    data = MUST.DataFrame(
        :teff=>[teff],:logg=>[logg],:feh=>[feh],
        :teff2=>[teff^2],:logg2=>[logg^2],:feh2=>[feh^2],
        :teff3=>[teff^3],:logg3=>[logg^3],:feh3=>[feh^3],
    )
    
    machines = cubic_regression_machines

    rho_bot = MLJ.predict_mean(machine(machines["bottom_rho"]), data[1:1, :]) |> first
    T_bot = MLJ.predict_mean(machine(machines["bottom_T"]), data[1:1, :]) |> first

    @info "Predicting boundary from regression: $(rho_bot), $(T_bot)"

    if !isnothing(offsets)
        @info "Adding offsets: $(offsets)"
        rho_bot+offsets[1], T_bot+offsets[2]
    else
        rho_bot, T_bot
    end
end

function extend_to_machine(model, eos; teff, logg, feh, extrapolation_offsets=nothing)
    rho_bot, t_bot = predict_from_machine(teff, logg, feh)
    mc = @optical(TSO.flip(deepcopy(model), depth=true), eos)

    model_extra = TSO.reverse_adiabatic_extrapolation(
        mc, exp10(rho_bot), exp10(t_bot), eos
    ) |> TSO.monotonic

    uniform_z = range(
        minimum(model_extra.z), maximum(model_extra.z), length=500
    ) |> collect

    TSO.flip(TSO.interpolate_to(model_extra, in_log=false, z=uniform_z), depth=true)
end

# ============================================================================
# Extrapolate model to bounds set by the user
# ============================================================================

function extrapolate_model(model_box, eos, tau_min, tau_max; regression=false, teff=model_box.parameter.teff, logg=model_box.parameter.logg, feh=0.0, extrapolation_offsets=nothing, outdir="")
    model = box2model(model_box)
    znew = TSO.rosseland_depth(eos, model)
    model.z .= znew
    TSO.optical_surface!(model)
    TSO.flip!(model)

    # recompute internal energy
    model.lnEi .= TSO.sample(eos, (:lnEi,), model.lnρ, model.lnT)[1]

    # if regression is given, interpolate from that point upwards until the model is hit
    model = if regression
        extend_to_machine(
            model, eos, extrapolation_offsets=extrapolation_offsets, teff=teff, logg=logg, feh=feh
        )
    else
        model
    end

    model = TSO.adiabatic_extrapolation(
        model, eos; τ_target=tau_max
    )
    ltau = range(tau_min, tau_max, length=500) |> collect
    model = TSO.flip(TSO.interpolate_to(model, τ=ltau, in_log=true), depth=true)

    TSO.flip!(model)
    av_path = abspath(joinpath(outdir, "inim.dat"))
    open(av_path, "w") do f
        MUST.writedlm(f, [model.z exp.(model.lnT) model.lnρ])
    end	

    model
end

# ============================================================================
# Construct namelist for M3DIS
# ============================================================================

function construct_namelist(ini_model, teff, logg=ini_model.logg; n_patches=[12,12,6], patch_size=20, atmos_size=[6.,6.,3.], shift_atmosphere_by=0.0, out_time=1.0, end_time=1000.0, outdir="", nml_name="m3dis.nml", kwargs...)
	# path of the model
	initial_path = abspath(joinpath(outdir, "inim.dat"))

	# compute length scale
	atmos_z_cgs = ini_model.z[end] - ini_model.z[1]
	l_cgs = atmos_z_cgs / atmos_size[3]

	# real physical size of the atmosphere
	atmos_size_cgs = atmos_size .* l_cgs

	# shift of the atmopshere to align optical surface
	z_top = ini_model.z[end]
	z_shift = atmos_size[3] / 2.0 - abs(z_top / l_cgs)

	# compute the velocity scale based on nu_max scaling relation
	pnew = teff^0.5 / exp10(logg)
    psun = 5777.0^0.5  / exp10(4.44)
    t_convective = 100 * pnew / psun
	v_cgs = l_cgs / t_convective

	# use the density at the optical surface as reference density
	mask = sortperm(ini_model.τ)
	rho_norm = MUST.linear_interpolation(
		MUST.Interpolations.deduplicate_knots!(
			log10.(ini_model.τ[mask]), move_knots=true
		),
		ini_model.lnρ[mask], 
		extrapolation_bc=MUST.Line()
	)(0.0) |> exp

	# use the internal enery at tau=-1 for the newton cooling
	lnee0 = MUST.linear_interpolation(
		MUST.Interpolations.deduplicate_knots!(
			log10.(ini_model.τ[mask]), move_knots=true
		),
		ini_model.lnEi[mask], 
		extrapolation_bc=MUST.Line()
	)(-1.0)
	
	zee0 = MUST.linear_interpolation(
		MUST.Interpolations.deduplicate_knots!(
			log10.(ini_model.τ[mask]), move_knots=true
		),
		ini_model.z[mask], 
		extrapolation_bc=MUST.Line()
	)(-1.0)

	# dummy namelist
	dummy_nml = MUST.StellarNamelist(abspath(joinpath(dirname(pathof(MUST)), "..", "initial_grids", "stellar_default.nml")))
	nml = deepcopy(dummy_nml)
	#nml = MUST.StellarNamelist()

	MUST.set!(
		nml,
		cartesian_params=(
            :size=>atmos_size, 
            :dims=>n_patches,
            :position=>[0,0,round(-z_shift, sigdigits=5)+shift_atmosphere_by]
        ),
		patch_params=(
			:n=>[patch_size, patch_size, patch_size],
		),
		scaling_params=(
            :l_cgs=>l_cgs, 
            :d_cgs=>rho_norm, 
            :v_cgs=>v_cgs
        ),
		stellar_params=(
            :g_cgs=>round(exp10(logg), digits=3), 
            :ee_min_cgs=>round(minimum(ini_model.lnEi), sigdigits=7), 
            :nz=>length(ini_model.z)-1, 
            :initial_path=>initial_path
        ),
		newton_params=(
			:position=>round(zee0/l_cgs, sigdigits=3),
			:ee0_cgs=>round(lnee0, sigdigits=7),
        ),
		gravity_params=(
            :constant=>-round(exp10(logg), digits=3),
        ),
		sc_rt_params=(  
            :rt_llc=>[-atmos_size[1]/2, -atmos_size[2]/2, -round(atmos_size[3]/2 + z_shift -shift_atmosphere_by, sigdigits=5)], 
            :rt_urc=>[ atmos_size[1]/2,  atmos_size[2]/2,  round(atmos_size[3]/2 - z_shift + shift_atmosphere_by, sigdigits=5)]
        ),
		boundary_params=(
			:target_teff=>teff,
		),
		io_params=(
            :out_time=>out_time,
            :end_time=>end_time
        ),
		eos_params=(
            :table_loc=>abspath(outdir),
		)
	)

    MUST.set!(nml; kwargs...)
    MUST.write(nml, joinpath(outdir, nml_name))
    MUST.write(nml, joinpath(outdir, "ininml.dat"))

	nml
end