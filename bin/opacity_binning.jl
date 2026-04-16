#!/usr/bin/env julia

using Pkg; Pkg.activate(joinpath(@__DIR__, "../examples"))
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

plt = matplotlib.pyplot
plt.switch_backend("Agg") 
TSO.USE_BINNING_THREADS[] = false

# ============================================================================
# CLI Parsing
# ============================================================================

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
            default = convert(Int, get(c, "iters", 500))
        "--n_neurons"
            help = "Number of neurons in the hidden layer of the neural network"
            arg_type = Int
            default = convert(Int, get(c, "n_neurons", 16))
        "--target_error"
            help = "Target error for the binning optimization"
            arg_type = Float64
            default = convert(Float64, get(c, "target_error", 0.02))
        # --- Output parameters (matching m1dis_star.jl logic) ---
        "--out_dir"
            help = "Output directory for the saved outputs (if --model is used, this is ignored and model dir is used)"
            default = get(c, "out_dir", "./models")
        "--model_name", "-n"
            help = "Output name of the model to save. If empty, auto-generated."
            default = get(c, "model_name", "")

        # --- Optional pre-computed model (overrides automatic computation) ---
        "--model"
            help = "Path to a pre-computed model HDF5 file. If provided, skips automatic atmosphere computation."
            arg_type = String
            default = get(c, "model", "")
    end

    args = parse_args(s)

    # Handle boolean flags that can come from config
    args["scattering"] = args["scattering"] || get(c, "scattering", false)

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

    # Get the final iteration result
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
    src_1d::Matrix{Float32}
    logg::Float64
    n_waves::Int
end

function run_1d_rt!(atm_obj, kappa, source)
    atm_obj.chi .= kappa
    atm_obj.B .= source
    M1DIS.update!(atm_obj)
    M1DIS.solve_gustafsson!(atm_obj, include_dT=false)
    return atm_obj.Q_rad, atm_obj.F_rad
end

function initialize_physics(model_box, eos_dir, n_bins)
    # "Initializing physics and thread-local pool..."
    eos_file = MUST.glob("*_eos_*.hdf5", eos_dir)[1]
    opa_file = MUST.glob("*_opacities_*.hdf5", eos_dir)[1]
    scat_file = MUST.glob("*_sopacities_*.hdf5", eos_dir)[1]
    
    eos_data = reload(eos_file) |> extended
    opa_data = reload(opa_file, mmap=true) |> extended
    scat_data = reload(scat_file, mmap=true) |> extended

    atm = M1DIS.Atmosphere(model_box, eos_data, opa_data, scattering=scat_data, downsample=2)

    n_waves = size(opa_data.opa.κ, 3)
    n_depths = length(atm.tau)
    wavelengths = wavelength(opa_data)
    logg = model_box.parameter.logg

    M1DIS.solve_approximate!(atm, include_dT=false)
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

    pool_size = Threads.nthreads() + 3
    atm_pool = Channel{M1DIS.Atmosphere}(pool_size)
    weights_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(atm_pool, deepcopy(atm_base))
        put!(weights_pool, zeros(Float32, n_waves, n_bins))
    end
    #@info "Allocated $(pool_size) thread-local atmospheres and weight buffers."

    rho = Float32.(atm.rho)
    temp = Float32.(atm.Temp)
    pgas = Float32.(atm.P_gas)
    lnr = Float32.(log.(atm.rho))
    lnt = Float32.(log.(atm.Temp))
    chi_1d, src_1d = transpose.(TSO.sample(eos_data, opa_data, (:κ, :src), lnr, lnt)) .|> collect

    return PhysicsContext(
        atm, atm_pool, weights_pool, Q_unbinned, Q_norm_factor, F_unbinned, F_norm_factor, wavelengths, opa_data.weights, 
        rho, temp, pgas, chi_1d, src_1d, logg, n_waves
    )
end

# ============================================================================
# Training Data & Pre-training 
# ============================================================================

function prepare_training_data(ctx::PhysicsContext, n_bins::Int, stripes::Bool)
    #@info "Preparing and normalizing data..."
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
    
    # Reshape (C, W) array to (W, C, N) tensor required by Flux 1D convolutions!
    X_cnn = reshape(collect(X_features'), (ctx.n_waves, size(X_features, 1), 1))
    
    return X_cnn, Y_targets
end

function pretrain_network(X_features, Y_targets, n_bins::Int, n_neurons::Int)
    #@info "Pre-training 1D CNN..."
    
    T = 5.0f0 # Temperature to keep softmax assignments fuzzy/soft
    model = Chain(
        Conv((10,), size(X_features, 2) => n_neurons, relu, pad=SamePad()),
        Conv((1,), n_neurons => n_bins),
        x -> dropdims(x, dims=3),
        x -> collect(transpose(x)),
        x -> x ./ T, 
        softmax
    )
    
    #optimizer = Flux.setup(Flux.OptimiserChain(Flux.WeightDecay(1e-3), Flux.Adam(0.05)), model)
    optimizer = Flux.setup(Flux.Adam(0.05), model)
    
    prog = Progress(500, desc="[Binning] Pre-training: ", color=:cyan)
    with_logger(NullLogger()) do
        for _ in 1:500
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
    base_model::Any
    base_weights_assign::AbstractMatrix{Float32}
end

function (objective::BinningObjective)(current_params)
    M = objective.restructure_model(Float32.(current_params))(objective.X_features)
    n_bins, n_waves = size(M)
    
    #=N_req = ceil(Int, 0.02 * n_waves)
    target_weight = 0.70f0
    empty_bin_penalty = 0.0f0
    
    for b in 1:n_bins
        row_vals = M[b, :] 
        
        sort!(row_vals, rev=true)
        
        for i in 1:N_req
            val = row_vals[i]
            if val < target_weight
                empty_bin_penalty += (target_weight - val)^2
            end
        end
    end
    
    final_penalty = empty_bin_penalty * objective.baseline_loss=#
    
    weights_assign = take!(objective.ctx.weights_pool)
    weights_assign .= transpose(M)
    kappa_box, src_box = TSO.advanced_binning_1d_quick(
        weights_assign, 
        objective.ctx.weights, 
        objective.ctx.wavelengths, 
        objective.ctx.rho, 
        objective.ctx.temp, 
        objective.ctx.pgas, 
        objective.ctx.chi_1d, 
        objective.ctx.src_1d, 
        logg=objective.ctx.logg
    )
    
    kappa_1d = transpose(dropdims(kappa_box, dims=1))
    src_1d = transpose(dropdims(src_box, dims=1))
    my_atm = take!(objective.ctx.atm_pool)

    local rt_loss
    try
        Q_binned, F_binned = run_1d_rt!(my_atm, kappa_1d, src_1d)
        err_Q = abs.((Q_binned .- objective.ctx.Q_unbinned) ./ objective.ctx.Q_norm_factor)
        rt_loss = maximum(err_Q)
    finally
        put!(objective.ctx.atm_pool, my_atm)
        put!(objective.ctx.weights_pool, weights_assign)
    end
    
    total_loss = rt_loss #+ final_penalty
    
    if total_loss < objective.best_loss[]
        objective.best_loss[] = total_loss
    end
    
    return total_loss
end

function optimize_weights(ctx::PhysicsContext, X_features, initial_params, restructure_model, iters::Int)    
    base_model = restructure_model(Float32.(initial_params))
    base_weights_assign = transpose(base_model(X_features))
    
    base_kappa, base_src = TSO.advanced_binning_1d_quick(
        base_weights_assign, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg
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
    
    objective_func = BinningObjective(restructure_model, X_features, ctx, baseline_loss, Ref(Inf), base_model, base_weights_assign)
    
    num_params = length(initial_params)
    lower_bounds = fill(-20., num_params)
    upper_bounds = fill(20., num_params)
    bounds = BoxConstraints(lower_bounds, upper_bounds)
    initial_params_64 = Float64.(initial_params)
    
    n_neurons = args["n_neurons"]
    bins = args["bins"]
    @info "Binning Optimization (Bins = $(bins), Iters = $(iters), Neurons = $(n_neurons))..."
    prog = Progress(iters, desc="[Binning] Optimizing: ", color=:magenta)
    
    iter_count = 0
    target_error = args["target_error"]
    
    cb = function(state)
        iter_count += 1
        current_best = objective_func.best_loss[]
        update!(prog, iter_count, showvalues=[(:Best_Loss, current_best)])
        
        if current_best < target_error
            @info "Target error of < $(target_error * 100)% reached. Stopping optimization."
            return true 
        end
        return false 
    end
    
    optimizer = CMAES(sigma0 = 1.0, lambda = 50)
    
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
    @show minimum(best_params), maximum(best_params)
    final_model = restructure_model(Float32.(best_params)) 
    return transpose(final_model(X_features))
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
    #@info "Saving results..."
    
    out_filename = "$(out_name)_assignment.txt"
    M1DIS.writedlm(out_filename, optimized_weights)
    @info "Saved assignment to $out_filename"

    kappa_box, src_box = TSO.advanced_binning_1d(
        optimized_weights, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg
    )
    
    final_atm = take!(ctx.atm_pool)
    try
        Q_final, F_final = run_1d_rt!(final_atm, transpose(dropdims(kappa_box, dims=1)), transpose(dropdims(src_box, dims=1)))
        err_Q = abs.((Q_final .- ctx.Q_unbinned) ./ ctx.Q_norm_factor)
        final_loss = maximum(err_Q)
        @info "Final Loss: $final_loss"

        plot_residuals(ctx, Q_final, F_final, final_atm.tau, out_name)
        plot_assignment(ctx, optimized_weights, n_bins, "$(out_name)_assignment.png")

    finally
        put!(ctx.atm_pool, final_atm)
    end

    #@info "Execution completed."
end

# ============================================================================
# Main
# ============================================================================

function main()
    Random.seed!(42) 

    args = parse_cli()
    n_bins = args["bins"]

    # Step 1: Resolve EoS directory
    eos_dir = resolve_eos_dir(args)

    # Step 2: Get the model — either from --model or by computing it
    model_box = if args["model"] != ""
        #@info "Loading pre-computed model from: $(args["model"])"
        Box(args["model"], mmap=false)
    else
        #@info "No --model provided. Computing atmosphere from stellar parameters..."
        compute_model(args, eos_dir)
    end

    # Step 3: Determine model name and output directory exactly like m1dis_star.jl
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

    out_dir = if args["model"] != ""
        dirname(abspath(args["model"]))
    else
        joinpath(abspath(args["out_dir"]), model_name)
    end

    if !isdir(out_dir)
        @info "Creating output directory: $out_dir"
        mkpath(out_dir)
    end
    
    out_prefix = joinpath(out_dir, model_name)

    @info "Starting M1DIS Binning..."
   
    stripes = true
    ctx = initialize_physics(model_box, eos_dir, n_bins)
    X_features, Y_targets = prepare_training_data(ctx, n_bins, stripes)
    initial_params, restructure_model = pretrain_network(X_features, Y_targets, n_bins, args["n_neurons"])
    
    # Plot initial pre-trained assignment and residuals
    initial_model = restructure_model(Float32.(initial_params))
    initial_weights = transpose(initial_model(X_features))
    plot_assignment(ctx, initial_weights, n_bins, "$(out_prefix)_pretrain_assignment.png")

    base_kappa, base_src = TSO.advanced_binning_1d(
        initial_weights, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg
    )
    pre_atm = take!(ctx.atm_pool)
    try
        Q_pre, F_pre = run_1d_rt!(pre_atm, transpose(dropdims(base_kappa, dims=1)), transpose(dropdims(base_src, dims=1)))
        plot_residuals(ctx, Q_pre, F_pre, pre_atm.tau, "$(out_prefix)_pretrain")
    finally
        put!(ctx.atm_pool, pre_atm)
    end

    optimized_weights = optimize_weights(ctx, X_features, initial_params, restructure_model, args["iters"])
    
    save_results_and_plot(ctx, optimized_weights, n_bins, out_prefix)
end

main()