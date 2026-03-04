#!/usr/bin/env julia

# Activate your project environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using M1DIS
using ArgParse
using TSO
using MUST
using TOML

function parse_commandline()
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

    s = ArgParseSettings(description = "M1DIS.jl Atmosphere Solver")

    @add_arg_table s begin
        "--config", "-c"
            help = "Path to a TOML configuration file. Command line arguments override these values."
            arg_type = String
            default = ""
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
            default = convert(Float64, get(c, "vmic", 0.0))
        "--maxiter"
            help = "Maximum number of iterations"
            arg_type = Int
            default = convert(Int, get(c, "maxiter", 20))
        "--alpha"
            help = "Mixing length parameter"
            arg_type = Float64
            default = convert(Float64, get(c, "alpha", 1.5))
        "--eos_dir"
            help = "Path to the directory containing the Equation of State and Opacity table files (required either via CLI or config)"
            default = get(c, "eos_dir", "")
        "--out_dir"
            help = "Output directory for the saved models"
            default = get(c, "out_dir", "./models")
        "--model_name"
            help = "Name of the model to save"
            default = get(c, "model_name", "m1dis_model")
        "--mini"
            help = "Ignore source function from the opacity table and recompute it on-the-fly."
            action = :store_true
        "--binned"
            help = "Signal that the opacity table uses binned opacities."
            action = :store_true
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
            default = convert(Int, get(c, "n_tau", 100))
        "--damping"
            help = "Damping parameter for the temperature updates."
            arg_type = Float64
            default = convert(Float64, get(c, "damping", 0.1))
        "--use_threads"
            help = "Use multi-threading for the RT."
            action = :store_true
    end

    args = parse_args(s)

    args["mini"] = args["mini"] || get(c, "mini", false)
    args["binned"] = args["binned"] || get(c, "binned", false)
    args["use_threads"] = args["use_threads"] || get(c, "use_threads", false)

    return args
end

function main()
    args = parse_commandline()
    
    if args["eos_dir"] == ""
        error("You must provide --eos_dir either via command line or in the config file.")
    end

    println("Starting M1DIS Execution...")
    println("Teff: $(args["teff"]) K, logg: $(args["logg"])")
    
    if args["binned"]
        eos_file = joinpath(args["eos_dir"], "eos_T.hdf5")
        eos500_file = joinpath(args["eos_dir"], "eos_T500.hdf5")
        opa_file = joinpath(args["eos_dir"], "binned_opacities_T.hdf5")
    else
        eos_file = MUST.glob("*_eos_*.hdf5", args["eos_dir"])[1]
        eos500_file = MUST.glob("*_eos500_*.hdf5", args["eos_dir"])[1]
        opa_file = MUST.glob("*_opacities_*.hdf5", args["eos_dir"])[1]
    end

    #println("Loading Equation of State from: $eos_file")
    eos_complete = TSO.reload(eos_file)
    
    #println("Loading Equation of State (500nm) from: $eos500_file")
    eos500_complete = TSO.reload(eos500_file)

    #println("Loading Opacity from: $opa_file")
    opa_complete = if args["mini"]
        TSO.reload(TSO.MiniOpacityTable, opa_file)
    else
        TSO.ExtendedOpacity(opa=TSO.reload(opa_file, mmap=!args["binned"]), binned=args["binned"])
    end

    use_threads = (!args["binned"]) || (args["use_threads"])

    println("================================================================================")
    println("===================== M1DIS.jl Atmosphere Solver ===============================")
    println("================================================================================")
    println("Computing atmosphere...")
    M1DIS.activate_timing!()
    M1DIS.start_timing!()
    result = atmosphere(
        T_eff = args["teff"],
        logg = args["logg"],
        #v_mac = args["vmic"],
        α_MLT = args["alpha"],
        maxiter = args["maxiter"],
        eos = eos_complete,
        opacity = opa_complete,
        damping = args["damping"],
        τ = 10.0 .^ range(args["tau_min"], args["tau_max"], length=args["n_tau"]),
        use_threads = use_threads,
        feutrier = true,
    )
    M1DIS.end_timing!()
    
    out_dir = args["out_dir"]
    if !isdir(out_dir)
        println("Creating output directory: $out_dir")
        mkpath(out_dir)
    end

    result = if !(typeof(result) <: AbstractArray)
        [result]
    else
        result
    end

    println("Saving model $(args["model_name"]) to $out_dir...")
    save!(
        result[end], args["model_name"]; 
        folder = out_dir, 
        vmic = args["vmic"], 
        logg = args["logg"],
        eos500 = eos500_complete
    )

    # save the iterations also
    if (!isdir(joinpath(out_dir, args["model_name"], "iterations")))
        mkpath(joinpath(out_dir, args["model_name"], "iterations"))
    else
        rm(joinpath(out_dir, args["model_name"], "iterations"), recursive=true)
        mkpath(joinpath(out_dir, args["model_name"], "iterations"))
    end
    for (i, r) in enumerate(result)
        save!(
            r, "iteration_$(i)"; 
            folder = joinpath(out_dir, args["model_name"], "iterations"), 
            vmic = args["vmic"], 
            logg = args["logg"],
            eos500 = eos500_complete
        )
    end
    
    println("M1DIS.jl finished.")
    println("================================================================================")
end

main()
