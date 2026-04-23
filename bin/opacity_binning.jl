#!/usr/bin/env julia

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using ArgParse
using M1DIS
using PythonPlot 
using Flux
using Flux.Losses: mse
using Optim
using Statistics
using TSO
using MUST
using TOML
using Logging
using ProgressMeter
using LaTeXStrings
using Evolutionary  
using Random
using Clustering
using MLJ
using MLJGLMInterface

plt = matplotlib.pyplot
plt.switch_backend("Agg") 

# ============================================================================
# CLI Parsing
# ============================================================================

function dict_to_symbol_tuples(d::Dict)
    pairs = Pair{Symbol, Any}[]
    for (k, v) in d
        if typeof(v) <: Dict
            push!(pairs, Symbol(k) => dict_to_symbol_tuples(v))
        else
            push!(pairs, Symbol(k) => v)
        end
    end
    return Tuple(pairs)
end

function parse_cli()
    # Pre-scan for config file
    config_file = ""
    for i in 1:length(ARGS)
        if ARGS[i] == "--config" || ARGS[i] == "-c"
            if i < length(ARGS)
                config_file = ARGS[i+1]
            end
            break
        end
    end

    c = Dict{String, Any}()
    if config_file != ""
        if !isfile(config_file)
            error("Configuration file not found: $config_file")
        end
        c = TOML.parsefile(config_file)
    end

    s = ArgParseSettings(description = "M1DIS.jl CMAES Binning with Automatic Model Computation")

    @add_arg_table s begin
        # --- Config file ---
        "--config", "-c"
            help = "Path to a TOML configuration file. Command line arguments override these values."
            arg_type = String
            default = ""

        # --- Stellar parameters (from m1dis_star.jl) ---
        "--teff"
            help = "Effective temperature (K)"
            arg_type = Float64
            default = convert(Float64, get(c, "teff", 5777.0))
        "--logg"
            help = "Surface gravity"
            arg_type = Float64
            default = convert(Float64, get(c, "logg", 4.44))
        "--vmic"
            help = "Microturbulence (km/s)"
            arg_type = Float64
            default = convert(Float64, get(c, "vmic", 1.0))
        "--feh"
            help = "Metallicity [Fe/H]"
            arg_type = Float64
            default = convert(Float64, get(c, "feh", 0.0))
        "--alpha"
            help = "Alpha enhancement."
            arg_type = Float64
            default = convert(Float64, get(c, "alpha", 0.0))
        "--composition"
            help = """
            Abundance of each element. All elements missing are set to the default values
            which are specified in the abund file (see eos config toml file).
            specify [X/Fe] composition here as e.g. "C_0.3O,O_0.2,Si_-0.3".
            """
            arg_type = String
            default = get(c, "composition", "")
        "--eos_config"
            help = "Path to the config file that contains the general EoS setup."
            default = get(c, "eos_config", "eos_config.toml")

        # --- Atmosphere solver parameters ---
        "--maxiter"
            help = "Maximum number of iterations for the atmosphere solver."
            arg_type = Int
            default = convert(Int, get(c, "maxiter", 20))
        "--alpha_MLT"
            help = "Mixing length parameter"
            arg_type = Float64
            default = convert(Float64, get(c, "alpha_MLT", 1.5))
        "--pbeta"
            help = "Parameter for turbulent pressure. 0.0 turns off turbulent pressure."
            arg_type = Float64
            default = convert(Float64, get(c, "pbeta", 0.0))
        "--tau_min"
            help = "Minimum optical depth to compute."
            arg_type = Float64
            default = convert(Float64, get(c, "tau_min", -6.0))
        "--tau_max"
            help = "Maximum optical depth to compute."
            arg_type = Float64
            default = convert(Float64, get(c, "tau_max", 2.0))
        "--n_tau"
            help = "Number of optical depth points."
            arg_type = Int
            default = convert(Int, get(c, "n_tau", 300))
        "--damping"
            help = "Damping parameter for the temperature updates."
            arg_type = Float64
            default = convert(Float64, get(c, "damping", 0.1))
        "--scattering"
            help = "Include scattering opacity separately from true absorption opacity."
            action = :store_true

        # --- Shared parameters ---
        "--eos_dir"
            help = "Path to the directory containing EoS and Opacity table files."
            arg_type = String
            default = get(c, "eos_dir", "")

        # --- Binning parameters ---
        "--bins", "-b"
            arg_type = Int
            default = convert(Int, get(c, "bins", 4))
        "--iters", "-i"
            arg_type = Int
            default = convert(Int, get(c, "iters", 1000))
        "--n_neurons"
            help = "Number of neurons in the hidden layer of the neural network"
            arg_type = Int
            default = convert(Int, get(c, "n_neurons", 32))
        "--n_walkers"
            help = """
            Number of walkers for the optimization. 
            If -1 they are automatically computed by the optimizer based on the number of model parameters. 
            Consider something like ~100. For fast execution it could be wise to take a multiple of the number of threads - 1, i.e. -t 31 --> --n_walkers=90. 
            """
            arg_type = Int
            default = convert(Int, get(c, "n_walkers", -1))
        "--convolutions"
            help = "Number of convolutions in the neural network"
            arg_type = Int
            default = convert(Int, get(c, "convolutions", 10))
        "--target_error"
            help = "Target error for the binning optimization"
            arg_type = Float64
            default = convert(Float64, get(c, "target_error", 0.035))
        # --- Output parameters (matching m1dis_star.jl logic) ---
        "--out_dir"
            help = "Output directory for the saved outputs (if --model is used, this is ignored and model dir is used)"
            default = get(c, "out_dir", "./models")
        "--model_name", "-n"
            help = "Output name of the model to save. If empty, auto-generated."
            default = get(c, "model_name", "")
        "--prefix"
            help = "Prefix added in front of model3D_name."
            arg_type = String
            default = get(c, "prefix", "")

        # --- Optional pre-computed model (overrides automatic computation) ---
        "--model"
            help = "Path to a pre-computed model HDF5 file. If provided, skips automatic atmosphere computation."
            arg_type = String
            default = get(c, "model", "")

        # --- Extrapolation Parameters ---
        "--extrapolate_tau_min"
            help = "Minimum optical depth for extrapolation."
            arg_type = Float64
            default = convert(Float64, get(c, "extrapolate_tau_min", -7.0))
        "--extrapolate_tau_max"
            help = "Maximum optical depth for extrapolation."
            arg_type = Float64
            default = convert(Float64, get(c, "extrapolate_tau_max", 7.0))
        "--extrapolate_regression"
            help = "Use regression algorithm for boundary condition extrapolation."
            action = :store_true
            
        # --- Namelist Base Parameters ---
        "--n_patches"
            help = "Number of patches in [x,y,z]. Provide as string, e.g. \"12,12,6\"."
            arg_type = String
        "--patch_size"
            help = "Size of each patch in cells."
            arg_type = Int
            default = convert(Int, get(c, "patch_size", 20))
        "--atmos_size"
            help = "Physical size of the atmosphere in [x,y,z]. Provide as string, e.g. \"6.0,6.0,3.0\"."
            arg_type = String
        "--shift_atmosphere_by"
            help = "Vertical shift of the atmosphere box."
            arg_type = Float64
            default = convert(Float64, get(c, "shift_atmosphere_by", 0.0))
            
        # --- Common Namelist Overrides ---
        "--out_time"
            help = "M3DIS output cadence (overrides io_params.out_time)."
            arg_type = Float64
            default = convert(Float64, get(c, "out_time", 1.0))
        "--end_time"
            help = "M3DIS simulation end time (overrides io_params.end_time)."
            arg_type = Float64
            default = convert(Float64, get(c, "end_time", 1000.0))
            
    end

    args = parse_args(s)

    # Handle boolean flags that can come from config
    args["scattering"] = args["scattering"] || get(c, "scattering", false)
    args["extrapolate_regression"] = args["extrapolate_regression"] || get(c, "extrapolate_regression", false)
    
    # Store config dict to access nested elements (e.g. [namelist])
    args["_config_obj"] = c

    # Pre-parse array strings
    parse_array(s::String, T) = [parse(T, x) for x in split(s, ',')]
    
    args["n_patches_parsed"] = if args["n_patches"] !== nothing
        parse_array(args["n_patches"], Int)
    else
        get(c, "n_patches", [12, 12, 6])
    end

    args["atmos_size_parsed"] = if args["atmos_size"] !== nothing
        parse_array(args["atmos_size"], Float64)
    else
        get(c, "atmos_size", [6.0, 6.0, 3.0])
    end

    # Pre-parse namelist kwargs
    namelist_dict = get(c, "namelist", Dict{String, Any}())
    args["namelist_kwargs"] = Dict{Symbol, Any}(Symbol(k) => dict_to_symbol_tuples(v) for (k,v) in namelist_dict)

    return args
end

# ============================================================================
# EoS / Opacity Resolution  (from m1dis_star.jl)
# ============================================================================

function resolve_eos_dir(args)
    if args["eos_dir"] != ""
        return args["eos_dir"]
    end

    eos_config_path = args["eos_config"]
    if !isabspath(eos_config_path)
        eos_config_path = joinpath(@__DIR__, eos_config_path)
    end
    if !isfile(eos_config_path)
        error("EoS configuration file not found at: $eos_config_path. Please provide a valid --eos_config or explicitly set --eos_dir.")
    end

    println("================================================================================")
    println("================== M1DIS.jl + TSO.jl Opacity Tables ============================")
    println("================================================================================")
    @info("Searching for tables with requested composition...")
    eos_c = TOML.parsefile(eos_config_path)

    try
        @import_tumult eos_c["m3d_dir"]
    catch e
        @warn "Could not import Multi3D. Please provide a valid path via --m3d_dir."
    end

    return M1DIS.get_or_compute_eos(
        args["feh"], 
        args["composition"];
        out_dir = get(eos_c, "opacity_tables_path", "data/opacities/"),
        modelatmosfolder = get(eos_c, "modelatmosfolder", "input_multi3d/test_opac_table/"),
        alpha = args["alpha"],
        abund = get(eos_c, "abund", "./input_multi3d/abund/abund_magg"),
        t_min = get(eos_c, "t_min", 1000.0), 
        t_max = get(eos_c, "t_max", 100000.0),
        rho_min = get(eos_c, "rho_min", 1e-18), 
        rho_max = get(eos_c, "rho_max", 1e-2),
        vmic = args["vmic"],
        lambda_min = get(eos_c, "lambda_min", 1000.0), 
        lambda_max = get(eos_c, "lambda_max", 200000.0),
        n_lambda = get(eos_c, "n_lambda", 100000), 
        n_t = get(eos_c, "n_t", 100), 
        n_rho = get(eos_c, "n_rho", 100),
        nnu = get(eos_c, "nnu", 32), 
        tmolim = get(eos_c, "tmolim", 100000.0),
        multi_threads = get(eos_c, "multi_threads", 20),
        linelist_dir = get(eos_c, "linelist_dir", "input_multi3d/master_linelists"),
        use_lambda_file = get(eos_c, "use_lambda_file", false), 
        lambda_file = get(eos_c, "lambda_file", "input_multi3d/flx_wavelengths_UV.vac"),
    )
end

# ============================================================================
# Automatic Atmosphere Computation  (from m1dis_star.jl)
# ============================================================================

function compute_model(args, eos_dir)
    println("================================================================================")
    println("===================== M1DIS.jl Atmosphere Solver ===============================")
    println("================================================================================")
    println("Computing atmosphere (Teff=$(args["teff"]) K, logg=$(args["logg"]))...")

    # Load EoS and opacity tables
    eos_file = MUST.glob("*_eos_*.hdf5", eos_dir)[1]
    opa_file = MUST.glob("*_opacities_*.hdf5", eos_dir)[1]

    eos_complete = TSO.reload(eos_file)
    opa_complete = TSO.ExtendedOpacity(opa=TSO.reload(opa_file, mmap=true), binned=false)

    scat_complete = if args["scattering"]
        scat_file = MUST.glob("*_sopacities_*.hdf5", eos_dir)[1]
        TSO.reload(TSO.MiniOpacityTable, scat_file)
    else
        nothing
    end

    M1DIS.activate_timing!()
    M1DIS.start_timing!()
    result = atmosphere(
        T_eff = args["teff"],
        logg = args["logg"],
        α_MLT = args["alpha_MLT"],
        pbeta = args["pbeta"],
        maxiter = args["maxiter"],
        eos = eos_complete,
        opacity = opa_complete,
        damping = args["damping"],
        τ = 10.0 .^ range(args["tau_min"], args["tau_max"], length=args["n_tau"]),
        use_threads = true,
        feutrier = true,
        scattering_opacity = scat_complete,
    )
    M1DIS.end_timing!()

    model_box = if typeof(result) <: AbstractArray
        result[end]
    else
        result
    end

    @info "Atmosphere computation complete."
    return model_box
end

# ============================================================================
# Physics Context for Binning  (from binning_CMAES.jl)
# ============================================================================

struct PhysicsContext
    atm::M1DIS.Atmosphere
    atm_pool::Channel{M1DIS.Atmosphere}
    weights_pool::Channel{Matrix{Float32}}
    Q_unbinned::Vector{Float64}
    Q_norm_factor::Float64
    F_unbinned::Vector{Float64}
    F_norm_factor::Float64
    wavelengths::Vector{Float32}
    weights::Vector{Float32}
    rho::Vector{Float32}
    temp::Vector{Float32}
    pgas::Vector{Float32}
    chi_1d::Matrix{Float32}
    chi_scat_1d::Matrix{Float32}
    src_1d::Matrix{Float32}
    logg::Float64
    n_waves::Int
    eos
    eos500
    opa
    scat
end

function run_1d_rt!(atm_obj, kappa, source)
    atm_obj.chi .= kappa
    atm_obj.B .= source
    M1DIS.update!(atm_obj)
    M1DIS.solve_gustafsson!(atm_obj, include_dT=false)
    return atm_obj.Q_rad, atm_obj.F_rad
end

function initialize_physics(model_box, eos_dir, n_bins)
    eos_file = MUST.glob("*_eos_*.hdf5", eos_dir)[1]
    eos500_file = MUST.glob("*_eos500_*.hdf5", eos_dir)[1]
    opa_file = MUST.glob("*_opacities_*.hdf5", eos_dir)[1]
    scat_file = MUST.glob("*_sopacities_*.hdf5", eos_dir)[1]
    
    eos_data = reload(eos_file) |> extended
    eos500_data = reload(eos500_file) |> extended
    opa_data = reload(opa_file, mmap=true) |> extended
    scat_data = reload(scat_file, mmap=true) |> extended

    atm = M1DIS.Atmosphere(model_box, eos_data, opa_data, scattering=scat_data, downsample=2)

    n_waves = size(opa_data.opa.κ, 3)
    n_depths = length(atm.tau)
    wavelengths = wavelength(opa_data)
    logg = model_box.parameter.logg

    M1DIS.solve_VEF!(atm, include_dT=false)
    Q_unbinned = deepcopy(atm.Q_rad)
    Q_norm_factor = maximum(abs.(Q_unbinned))
    F_unbinned = deepcopy(atm.F_rad)
    F_norm_factor = maximum(abs.(F_unbinned))

    atm_base = M1DIS.Atmosphere(; 
        T_eff=atm.T_eff, z=atm.z, tau=atm.tau, rho=atm.rho, Temp=atm.Temp, P_gas=atm.P_gas, 
        mu=atm.mu, w_mu=atm.w_mu, chi_ref=atm.chi_ref, 
        chi=zeros(Float64, n_bins, n_depths), B=zeros(Float64, n_bins, n_depths), 
        dBdT=zeros(Float64, n_bins, n_depths), dchidT=zeros(Float64, n_bins, n_depths)
    )

    pool_size = Threads.nthreads() + 5
    atm_pool = Channel{M1DIS.Atmosphere}(pool_size)
    weights_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(atm_pool, deepcopy(atm_base))
        put!(weights_pool, zeros(Float32, n_waves, n_bins))
    end

    rho = Float32.(atm.rho)
    temp = Float32.(atm.Temp)
    pgas = Float32.(atm.P_gas)
    lnr = Float32.(log.(atm.rho))
    lnt = Float32.(log.(atm.Temp))
    chi_1d, src_1d = transpose.(TSO.sample(eos_data, opa_data, (:κ, :src), lnr, lnt)) .|> collect
    chi_scat_1d, = transpose.(TSO.sample(eos_data, scat_data, (:κ, ), lnr, lnt)) .|> collect

    return PhysicsContext(
        atm, atm_pool, weights_pool, Q_unbinned, Q_norm_factor, F_unbinned, F_norm_factor, wavelengths, opa_data.weights, 
        rho, temp, pgas, chi_1d, chi_scat_1d, src_1d, logg, n_waves, eos_data, eos500_data, opa_data, scat_data
    )
end

# ============================================================================
# Training Data & Pre-training 
# ============================================================================

function prepare_training_data(ctx::PhysicsContext, n_bins::Int, stripes::Bool)
    X_features = Base.convert(
        Matrix{Float32}, 
        vcat(
            log10.(ctx.wavelengths)', 
            M1DIS.formation_height(ctx.atm, closest=true)', 
            M1DIS.formation_source_function(ctx.atm, closest=true)'
        ) |> collect
    )

    feature_means = mean(X_features, dims=2)
    feature_stds = std(X_features, dims=2)    
    feature_stds[feature_stds .== 0] .= 1.0f0 
    X_features .= (X_features .- feature_means) ./ feature_stds
    
    Y_targets = zeros(Float32, n_bins, ctx.n_waves)

    data = stripes ? X_features[2:end, :] : X_features
    clusters = kmeans(Matrix{Float64}(data), n_bins; maxiter=200)

    for i in 1:ctx.n_waves
        Y_targets[clusters.assignments[i], i] = 1.0f0
    end
    
    # Reshape (C, W) array to (W, C, N)
    X_cnn = reshape(collect(X_features'), (ctx.n_waves, size(X_features, 1), 1))
    
    return X_cnn, Y_targets
end

function pretrain_network(X_features, Y_targets, n_bins::Int, n_neurons::Int, n_conv::Int)    
    T = 5.0f0
    model = Chain(
        Conv((n_conv,), size(X_features, 2) => n_neurons, relu, pad=SamePad()),
        Conv((1,), n_neurons => n_bins),
        x -> dropdims(x, dims=3),
        x -> collect(transpose(x)),
        x -> x ./ T, 
        softmax
    )
    
    optimizer = Flux.setup(Flux.Adam(0.05), model)
    
    prog = Progress(2000, desc="[Binning] Pre-training: ", color=:cyan)
    with_logger(NullLogger()) do
        for _ in 1:2000
            Flux.train!((m, x, y) -> mse(m(x), y), model, [(X_features, Y_targets)], optimizer)
            next!(prog)
        end
    end
    
    return Flux.destructure(model)
end

# ============================================================================
# CMAES Optimization 
# ============================================================================

struct BinningObjective
    restructure_model::Any
    X_features::Array{Float32, 3}
    ctx::PhysicsContext
    baseline_loss::Float64
    best_loss::Base.RefValue{Float64}
    best_Q_loss::Base.RefValue{Float64}
    best_F_loss::Base.RefValue{Float64}
    base_model::Any
    base_weights_assign::AbstractMatrix{Float32}
end

function (objective::BinningObjective)(current_params)
    M = objective.restructure_model(Float32.(current_params))(objective.X_features)
    weights_assign = take!(objective.ctx.weights_pool)
    weights_assign .= transpose(M)
    clean_weights!(weights_assign)
    
    n_waves, n_bins = size(weights_assign)
    bin_masses = sum(weights_assign, dims=1)
    
    min_allowed_mass = n_waves * 0.02 
    penalty = 0.0
    for m in bin_masses
        if m < min_allowed_mass
            penalty += ((min_allowed_mass - m) / min_allowed_mass)^2 
        end
    end
    
    kappa_box, src_box = TSO.advanced_binning_1d_quick(
        weights_assign, 
        objective.ctx.weights, 
        objective.ctx.wavelengths, 
        objective.ctx.rho, 
        objective.ctx.temp, 
        objective.ctx.pgas, 
        objective.ctx.chi_1d, 
        objective.ctx.src_1d, 
        logg=objective.ctx.logg,
        κ_scat_1d=objective.ctx.chi_scat_1d
    )
    
    kappa_1d = transpose(dropdims(kappa_box, dims=1))
    src_1d = transpose(dropdims(src_box, dims=1))
    my_atm = take!(objective.ctx.atm_pool)
    
    local rt_loss, F_loss

    try
        Q_binned, F_binned = run_1d_rt!(my_atm, kappa_1d, src_1d)
        err_Q = abs.((Q_binned .- objective.ctx.Q_unbinned) ./ objective.ctx.Q_norm_factor)
        err_F = abs.((F_binned .- objective.ctx.F_unbinned) ./ objective.ctx.F_unbinned)
        rt_loss = maximum(err_Q)
        F_loss = maximum(err_F)
    finally
        put!(objective.ctx.atm_pool, my_atm)
        put!(objective.ctx.weights_pool, weights_assign)
    end
    
    total_loss = sqrt(2.0 * rt_loss^2 + F_loss^2) + (100.0 * penalty)
    
    if total_loss < objective.best_loss[]
        objective.best_loss[] = total_loss
        objective.best_Q_loss[] = rt_loss
        objective.best_F_loss[] = F_loss
    end
    
    return total_loss
end

function clean_weights!(weights::AbstractMatrix{T}; digits::Int=4, tol::T=T(10.0^(-(digits + 2)))) where {T <: AbstractFloat}   
    rows, cols = size(weights)
    @inbounds for i in 1:rows
        max_val = typemin(T)
        max_idx = 1
        row_sum = zero(T)
        
        for j in 1:cols
            val = weights[i, j]
            
            if val > max_val
                max_val = val
                max_idx = j
            end
            
            rounded_val = round(val, digits=digits)
            weights[i, j] = rounded_val
            row_sum += rounded_val
        end
        
        diff = one(T) - row_sum
        if abs(diff) > tol
            weights[i, max_idx] += diff
        end
    end
    
    return weights
end

function optimize_weights(ctx::PhysicsContext, X_features, initial_params, restructure_model, iters::Int, target_error, n_walkers)    
    base_model = restructure_model(Float32.(initial_params))
    base_weights_assign = transpose(base_model(X_features))
    clean_weights!(base_weights_assign)

    base_kappa, base_src = TSO.advanced_binning_1d_quick(
        base_weights_assign, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg, κ_scat_1d=ctx.chi_scat_1d
    )
    
    base_atm = take!(ctx.atm_pool)
    local baseline_loss
    try
        Q_base, F_base = run_1d_rt!(base_atm, transpose(dropdims(base_kappa, dims=1)), transpose(dropdims(base_src, dims=1)))
        err_Q = abs.((Q_base .- ctx.Q_unbinned) ./ ctx.Q_norm_factor)
        baseline_loss = maximum(err_Q)
    finally
        put!(ctx.atm_pool, base_atm)
    end
    
    objective_func = BinningObjective(restructure_model, X_features, ctx, baseline_loss, Ref(Inf), Ref(Inf), Ref(Inf), base_model, base_weights_assign)
    
    num_params = length(initial_params)
    lower_bounds = fill(-25., num_params)
    upper_bounds = fill(25., num_params)
    bounds = BoxConstraints(lower_bounds, upper_bounds)
    initial_params_64 = Float64.(initial_params)
    
    @info "⌛ Binning Optimization..."
    prog = Progress(iters, desc="[Binning] Optimizing: ", color=:magenta)
    
    iter_count = 0    
    cb = function(state)
        iter_count += 1
        current_best = objective_func.best_Q_loss[]
        update!(
            prog, 
            iter_count, 
            showvalues=[
                (Symbol("Q loss (max)"), objective_func.best_Q_loss[]), 
                (Symbol("F loss (max)"), objective_func.best_F_loss[]), 
                (Symbol("total loss (combined)"), objective_func.best_loss[]),
                (Symbol("target error (Q max)"), target_error)
            ]
        )
        
        if current_best < target_error
            return true 
        end
        return false 
    end
    
    #optimizer = n_walkers > 0 ? CMAES(λ = n_walkers) : CMAES()
    optimizer = n_walkers > 0 ? CMAES(λ = n_walkers, μ = floor(Int, n_walkers / 2)) : CMAES()
    
    options = Evolutionary.Options(
        iterations = iters,
        callback = cb,
        parallelization = :thread,
        show_trace = false,
        reltol = 1e-5,          
        successive_f_tol = 100    
    )
    
    res = Evolutionary.optimize(
        objective_func, 
        bounds,
        initial_params_64, 
        optimizer, 
        options
    )
    
    finish!(prog) 
    
    best_params = Evolutionary.minimizer(res)
    final_model = restructure_model(Float32.(best_params)) 
    w = transpose(final_model(X_features))
    clean_weights!(w)
    return w
end

# ============================================================================
# Results & Plotting 
# ============================================================================

function plot_residuals(ctx::PhysicsContext, Q_binned, F_binned, tau, out_name::String)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(log10.(tau), (Q_binned .- ctx.Q_unbinned) ./ ctx.Q_norm_factor)
    ax1.set_xlabel(L"\log_{10}(\tau)")
    ax1.set_ylabel(L"\Delta Q_{rad} / Q_{max}")
    fig1.savefig("$(out_name)_residuals_Q.png")
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(log10.(tau), (F_binned .- ctx.F_unbinned) ./ ctx.F_unbinned)
    ax3.set_xlabel(L"\log_{10}(\tau)")
    ax3.set_ylabel(L"\Delta F_{rad} / F_{rad}")
    fig3.savefig("$(out_name)_residuals_F.png")
    
    plt.close(fig1)
    plt.close(fig3)
end

function plot_assignment(ctx::PhysicsContext, weights, n_bins::Int, filename::String)
    fig, ax = plt.subplots(figsize=(10, 5))
    log_waves = log10.(ctx.wavelengths)
    x_min, x_max = minimum(log_waves), maximum(log_waves)
    y_min, y_max = 0.5, n_bins + 0.5 
    im = ax.imshow(
        transpose(weights), 
        extent=[x_min, x_max, y_min, y_max], 
        origin="lower", 
        aspect="auto", 
        cmap="viridis",
        interpolation="nearest"
    )
    ax.set_xlabel(L"\log_{10}(\lambda)")
    ax.set_ylabel("Bin Index")
    fig.colorbar(im, ax=ax, label="Bin Weight")
    fig.savefig(filename)
    plt.close(fig)
end

function save_results_and_plot(ctx::PhysicsContext, optimized_weights, n_bins::Int, out_name::String)    
    out_filename = "$(out_name)_assignment.txt"
    M1DIS.writedlm(out_filename, optimized_weights)

    kappa_box, src_box = TSO.advanced_binning_1d_quick(
        optimized_weights, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg, κ_scat_1d=ctx.chi_scat_1d
    )
    
    final_atm = take!(ctx.atm_pool)
    try
        Q_final, F_final = run_1d_rt!(final_atm, transpose(dropdims(kappa_box, dims=1)), transpose(dropdims(src_box, dims=1)))
        err_Q = abs.((Q_final .- ctx.Q_unbinned) ./ ctx.Q_norm_factor)
        final_loss = maximum(err_Q)
        @info "Final Q max error: $(final_loss*100) %"

        plot_residuals(ctx, Q_final, F_final, final_atm.tau, out_name)
        plot_assignment(ctx, optimized_weights, n_bins, "$(out_name)_assignment.png")

    finally
        put!(ctx.atm_pool, final_atm)
    end
end

# ============================================================================
# Compute initial condition for M3DIS directly from binned opacities
# ============================================================================

function save_table(aos, aos500,binned_opacities, model, target_dir)
    save(binned_opacities.opacities, joinpath(target_dir, "binned_opacities_T.hdf5"))
    save(aos.eos, joinpath(target_dir, "eos_T.hdf5"))
    save(aos500.eos, joinpath(target_dir, "eos500_T.hdf5"))

	lnEi, = TSO.sample(aos, (:lnEi,), model[:logd], model[:logT])
	lnEimin, lnEimax = TSO.get_e_limit(aos.eos, lnEi, 1.5)
	eosE, opaE = TSO.remap_T_to_E(
		aos,
		binned_opacities.opacities, 
		upsample=2048,
		lnEimin=lnEimin,
		lnEimax=lnEimax
	)
	
	for_dispatch(eosE, opaE.κ, opaE.src, ones(eltype(opaE.src), size(opaE.src)...), target_dir)

	save(opaE, joinpath(target_dir, "binned_opacities.hdf5"))
    save(eosE, joinpath(target_dir, "eos.hdf5"))
	
    return target_dir
end

include("initial_condition.jl")

# ============================================================================
# Main
# ============================================================================

function main()
    Random.seed!(42) 

    # multi-threading used in the optimizer internally, so turn it off elsewhere
    TSO.USE_BINNING_THREADS[] = false

    args = parse_cli()
    n_bins = args["bins"]
    eos_dir = resolve_eos_dir(args)

    model_box = if args["model"] != ""
        Box(args["model"], mmap=false)
    else
        compute_model(args, eos_dir)
    end

    model_name = if args["model"] != ""
        replace(basename(args["model"]), r"\.hdf5$"i => "")
    elseif args["model_name"] == ""
        a = args["alpha"]
        z = args["feh"]
        v = args["vmic"]
        "p$(args["teff"])_g$(args["logg"])_z$(z)_a$(a)_vmic$(v)"
    else
        args["model_name"]
    end

    model3D_name = if args["model"] != ""
        replace(basename(args["model"]), r"\.hdf5$"i => "")
    elseif args["model_name"] == ""
        a = args["alpha"]
        z = args["feh"]
        v = args["vmic"]
        "t$(round(args["teff"]/10, digits=2))g$(round(args["logg"]*10, digits=2))m$(round(z, digits=4))_a$(a)"
    else
        args["model_name"]
    end
    
    if args["prefix"] != ""
        model3D_name = args["prefix"] * "_" * model3D_name
    end

    out_dir = if args["model"] != ""
        dirname(abspath(args["model"]))
    else
        joinpath(abspath(args["out_dir"]), model_name)
    end

    # make a new sub-dir for binning related things
    out_dir = joinpath(out_dir, "opacity_binning")

    if !isdir(out_dir)
        @info "Creating output directory: $out_dir"
        mkpath(out_dir)
    end
    
    out_prefix = joinpath(out_dir, model_name)

    println("================================================================================")
    println("=========================== M1DIS.jl Binning ===================================")
    println("================================================================================")
    @info "Starting M1DIS Binning..."
   
    stripes = true
    ctx = initialize_physics(model_box, eos_dir, n_bins)
    X_features, Y_targets = prepare_training_data(ctx, n_bins, stripes)
    initial_params, restructure_model = pretrain_network(X_features, Y_targets, n_bins, args["n_neurons"], args["convolutions"])
    
    initial_model = restructure_model(Float32.(initial_params))
    initial_weights = transpose(initial_model(X_features))
    plot_assignment(ctx, initial_weights, n_bins, "$(out_prefix)_pretrain_assignment.png")

    base_kappa, base_src = TSO.advanced_binning_1d_quick(
        initial_weights, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg, κ_scat_1d=ctx.chi_scat_1d
    )
    pre_atm = take!(ctx.atm_pool)
    try
        Q_pre, F_pre = run_1d_rt!(pre_atm, transpose(dropdims(base_kappa, dims=1)), transpose(dropdims(base_src, dims=1)))
        plot_residuals(ctx, Q_pre, F_pre, pre_atm.tau, "$(out_prefix)_pretrain")
    finally
        put!(ctx.atm_pool, pre_atm)
    end

    optimized_weights = optimize_weights(ctx, X_features, initial_params, restructure_model, args["iters"], args["target_error"], args["n_walkers"])
    save_results_and_plot(ctx, optimized_weights, n_bins, out_prefix)

    # opacity bins are assigned, now bin the full table
    TSO.USE_BINNING_THREADS[] = true

    # compute the final binning for the entire table
    binned_opacities = TSO.advanced_binning(
        optimized_weights, ctx.weights, ctx.eos.eos, opacity(ctx.opa), opacity(ctx.scat), logg=ctx.logg
    )

    # store the results and convert to dispatch format
    save_table(ctx.eos.eos, ctx.eos500.eos, binned_opacities, model_box, out_dir)
    @info "Binning complete."
    
    extrapolate_offsets = get(args["_config_obj"], "extrapolate_offsets", nothing)
    
    # extrapolate the model to the desired optical depth range
    model_extra = extrapolate_model(
        model_box, ctx.eos.eos.eos, args["extrapolate_tau_min"], args["extrapolate_tau_max"], 
        regression=args["extrapolate_regression"], 
        extrapolation_offsets=extrapolate_offsets,
        outdir=out_dir
    )

    # construct namelist for M3DIS
    nml_name = model3D_name*".nml"
    nml = construct_namelist(
        model_extra, 
        model_box.parameter.teff, 
        model_box.parameter.logg,
        n_patches=args["n_patches_parsed"],
        patch_size=args["patch_size"],
        atmos_size=args["atmos_size_parsed"],
        shift_atmosphere_by=args["shift_atmosphere_by"],
        out_time=args["out_time"],
        end_time=args["end_time"],
        outdir=out_dir, 
        nml_name=nml_name;
        args["namelist_kwargs"]...
    )
    @info "Initial condition for M3DIS prepared."
    @info "All results stored in $out_dir."
    @info "M1DIS complete."
end

main()