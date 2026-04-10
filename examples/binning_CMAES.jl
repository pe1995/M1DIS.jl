#!/usr/bin/env julia

using Pkg; Pkg.activate(".")
using ArgParse
using M1DIS
using PythonPlot 
using Flux
using Flux.Losses: mse
using Optim
using Statistics
using TSO
using MUST
using Logging
using ProgressMeter
using LaTeXStrings
using Evolutionary  

plt = matplotlib.pyplot
plt.switch_backend("Agg") 
TSO.USE_BINNING_THREADS[] = false

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--bins", "-b"; arg_type = Int; default = 4
        "--iters", "-i"; arg_type = Int; default = 200
        "--eos_dir"; arg_type = String; default = "../../opacity_tables/magg_m0_a0_vmic1_v3.5/"
        "--model"; arg_type = String; default = "../models/p5777.0_g4.44_z0.0_a0.0_vmic1.0/p5777.0_g4.44_z0.0_a0.0_vmic1.0.hdf5"
        "--out_name"; arg_type = String; default = "test_binning"
    end
    return parse_args(s)
end

struct PhysicsContext
    atm::M1DIS.Atmosphere
    atm_pool::Channel{M1DIS.Atmosphere}
    Q_unbinned::Vector{Float64}
    Q_norm_factor::Float64
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
    return atm_obj.Q_rad
end

function initialize_physics(args)
    @info "Initializing physics and thread-local pool..."
    eos_file = MUST.glob("*_eos_*.hdf5", args["eos_dir"])[1]
    opa_file = MUST.glob("*_opacities_*.hdf5", args["eos_dir"])[1]
    scat_file = MUST.glob("*_sopacities_*.hdf5", args["eos_dir"])[1]
    
    eos_data = reload(eos_file) |> extended
    opa_data = reload(opa_file, mmap=true) |> extended
    scat_data = reload(scat_file, mmap=true) |> extended
    model_box = Box(args["model"], mmap=false)

    atm = M1DIS.Atmosphere(model_box, eos_data, opa_data, scattering=scat_data, downsample=2)

    n_waves = size(opa_data.opa.κ, 3)
    n_bins = args["bins"]
    n_depths = length(atm.tau)
    wavelengths = wavelength(opa_data)
    logg = model_box.parameter.logg

    M1DIS.solve_approximate!(atm, include_dT=false)
    Q_unbinned = deepcopy(atm.Q_rad)
    Q_norm_factor = maximum(abs.(Q_unbinned))

    atm_base = M1DIS.Atmosphere(; 
        T_eff=atm.T_eff, z=atm.z, tau=atm.tau, rho=atm.rho, Temp=atm.Temp, P_gas=atm.P_gas, 
        mu=atm.mu, w_mu=atm.w_mu, chi_ref=atm.chi_ref, 
        chi=zeros(Float64, n_bins, n_depths), B=zeros(Float64, n_bins, n_depths), 
        dBdT=zeros(Float64, n_bins, n_depths), dchidT=zeros(Float64, n_bins, n_depths)
    )

    pool_size = Threads.nthreads() + 5
    atm_pool = Channel{M1DIS.Atmosphere}(pool_size)
    for _ in 1:pool_size
        put!(atm_pool, deepcopy(atm_base))
    end
    @info "Allocated $(pool_size) thread-local atmospheres."

    rho = Float32.(atm.rho)
    temp = Float32.(atm.Temp)
    pgas = Float32.(atm.P_gas)
    lnr = Float32.(log.(atm.rho))
    lnt = Float32.(log.(atm.Temp))
    chi_1d, src_1d = transpose.(sample(eos_data, opa_data, (:κ, :src), lnr, lnt)) .|> collect

    return PhysicsContext(
        atm, atm_pool, Q_unbinned, Q_norm_factor, wavelengths, opa_data.weights, 
        rho, temp, pgas, chi_1d, src_1d, logg, n_waves
    )
end

function prepare_training_data(ctx::PhysicsContext, n_bins::Int)
    @info "Preparing and normalizing data..."
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
    chunk_size = cld(ctx.n_waves, n_bins)
    height_indices = sortperm(X_features[2, :])
    
    for b in 1:n_bins
        idx = height_indices[((b-1)*chunk_size + 1) : min(b*chunk_size, ctx.n_waves)]
        Y_targets[b, idx] .= 1.0
    end
    
    return X_features, Y_targets
end

function pretrain_network(X_features, Y_targets, n_bins::Int)
    @info "Pre-training..."
    model = Chain(Dense(3 => 4, relu), Dense(4 => n_bins), softmax)
    optimizer = Flux.setup(Flux.Adam(0.05), model)
    
    prog = Progress(1000, desc="[Binning] Pre-training: ", color=:cyan)
    with_logger(NullLogger()) do
        for _ in 1:1000
            Flux.train!((m, x, y) -> mse(m(x), y), model, [(X_features, Y_targets)], optimizer)
            next!(prog)
        end
    end
    
    return Flux.destructure(model)
end

struct BinningObjective
    restructure_model::Any
    X_features::Matrix{Float32}
    ctx::PhysicsContext
    baseline_loss::Float64
end

function (objective::BinningObjective)(current_params)
    model = objective.restructure_model(Float32.(current_params))
    weights_assign = transpose(model(objective.X_features))
    
    #=bin_totals = sum(weights_assign, dims=1)
    min_fraction = 0.02 
    threshold = min_fraction * objective.ctx.n_waves
    k = 0.05 
    shifted_exp = sum(exp.(k .* (threshold .- bin_totals)))
    empty_bin_penalty = shifted_exp * (objective.baseline_loss * 10.0)=#

    steepness = 30.0
    ownership_threshold = 0.70
    strong_ownership = 1.0 ./ (1.0 .+ exp.(.-steepness .* (weights_assign .- ownership_threshold)))
    strong_counts = sum(strong_ownership, dims=1)
    min_fraction = 0.01 
    population_threshold = min_fraction * objective.ctx.n_waves
    shortfall = max.(0.0, population_threshold .- strong_counts)
    empty_bin_penalty = sum(shortfall .^ 2) * (objective.baseline_loss * 10.0)
    
    kappa_box, src_box = TSO.advanced_binning_1d(
        weights_assign, objective.ctx.weights, objective.ctx.wavelengths, objective.ctx.rho, objective.ctx.temp, objective.ctx.pgas, 
        objective.ctx.chi_1d, objective.ctx.src_1d, logg=objective.ctx.logg
    )
    
    kappa_1d = transpose(dropdims(kappa_box, dims=1))
    src_1d = transpose(dropdims(src_box, dims=1))
    
    my_atm = take!(objective.ctx.atm_pool)
    local rt_loss
    try
        Q_binned = run_1d_rt!(my_atm, kappa_1d, src_1d)
        rt_loss = mean(((Q_binned .- objective.ctx.Q_unbinned) ./ objective.ctx.Q_norm_factor) .^ 2)
    finally
        put!(objective.ctx.atm_pool, my_atm)
    end
    
    return rt_loss + empty_bin_penalty
end

function optimize_weights(ctx::PhysicsContext, X_features, initial_params, restructure_model, iters::Int)
    @info "Calculating static baseline loss from pre-trained weights..."
    
    base_model = restructure_model(Float32.(initial_params))
    base_weights_assign = transpose(base_model(X_features))
    
    base_kappa, base_src = TSO.advanced_binning_1d(
        base_weights_assign, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg
    )
    
    base_atm = take!(ctx.atm_pool)
    local baseline_loss
    try
        Q_base = run_1d_rt!(base_atm, transpose(dropdims(base_kappa, dims=1)), transpose(dropdims(base_src, dims=1)))
        baseline_loss = mean(((Q_base .- ctx.Q_unbinned) ./ ctx.Q_norm_factor) .^ 2)
    finally
        put!(ctx.atm_pool, base_atm)
    end
    
    objective_func = BinningObjective(restructure_model, X_features, ctx, baseline_loss)
    
    num_params = length(initial_params)
    lower_bounds = fill(-5.0, num_params)
    upper_bounds = fill(5.0, num_params)
    bounds = BoxConstraints(lower_bounds, upper_bounds)
    initial_params_64 = Float64.(initial_params)
    
    prog = Progress(iters, desc="[Binning] Optimizing: ", color=:magenta)
    
    iter_count = 0
    cb = function(state)
        iter_count += 1
        update!(prog, iter_count)
        return false 
    end
    
    optimizer = CMAES() 
    
    options = Evolutionary.Options(
        iterations = iters,
        callback = cb,
        parallelization = :thread,
        show_trace = false,
        reltol = 1e-4,          
        successive_f_tol = 5    
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
    return transpose(final_model(X_features))
end

function save_results_and_plot(ctx::PhysicsContext, optimized_weights, n_bins::Int, out_name::String)
    @info "Saving results..."
    
    out_filename = "$(out_name)_assignment.txt"
    M1DIS.writedlm(out_filename, optimized_weights)
    @info "Saved assignment to $out_filename"

    kappa_box, src_box = TSO.advanced_binning_1d(
        optimized_weights, ctx.weights, ctx.wavelengths, ctx.rho, ctx.temp, ctx.pgas, 
        ctx.chi_1d, ctx.src_1d, logg=ctx.logg
    )
    
    final_atm = take!(ctx.atm_pool)
    try
        Q_final = run_1d_rt!(final_atm, transpose(dropdims(kappa_box, dims=1)), transpose(dropdims(src_box, dims=1)))
        final_loss = mean(((Q_final .- ctx.Q_unbinned)) .^ 2)
        @info "Final Loss: $final_loss"

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(log10.(final_atm.tau), (Q_final .- ctx.Q_unbinned) ./ ctx.Q_norm_factor)
        ax1.set_xlabel(L"\log_{10}(\tau)")
        ax1.set_ylabel(L"\Delta Q_{rad} / Q_{max}")
        fig1.savefig("$(out_name)_residuals.png")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        log_waves = log10.(ctx.wavelengths)
        x_min, x_max = minimum(log_waves), maximum(log_waves)
        y_min, y_max = 0.5, n_bins + 0.5 
        im = ax2.imshow(
            transpose(optimized_weights), 
            extent=[x_min, x_max, y_min, y_max], 
            origin="lower", 
            aspect="auto", 
            cmap="viridis",
            interpolation="nearest"
        )
        ax2.set_xlabel(L"\log_{10}(\lambda)")
        ax2.set_ylabel("Bin Index")
        fig2.colorbar(im, ax=ax2, label="Bin Weight")
        fig2.savefig("$(out_name)_assignment.png")

    finally
        put!(ctx.atm_pool, final_atm)
    end

    @info "Execution completed."
end

function main()
    args = parse_cli()
    n_bins = args["bins"]

    @info "Starting M1DIS Binning"
    @info "Optimization configuration: Bins=$n_bins, Iterations=$(args["iters"])"

    ctx = initialize_physics(args)
    X_features, Y_targets = prepare_training_data(ctx, n_bins)
    initial_params, restructure_model = pretrain_network(X_features, Y_targets, n_bins)
    optimized_weights = optimize_weights(ctx, X_features, initial_params, restructure_model, args["iters"])
    
    save_results_and_plot(ctx, optimized_weights, n_bins, args["out_name"])
end

main()