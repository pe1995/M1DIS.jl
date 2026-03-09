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
            default = get(c, "eos_config", "bin/eos_config.toml")
        "--maxiter"
            help = "Maximum number of iterations"
            arg_type = Int
            default = convert(Int, get(c, "maxiter", 20))
        "--alpha_MLT"
            help = "Mixing length parameter"
            arg_type = Float64
            default = convert(Float64, get(c, "alpha_MLT", 1.5))
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
        "--scattering"
            help = "Include scattering opacity separately from true absorption opacity."
            action = :store_true
    end

    args = parse_args(s)

    args["mini"] = args["mini"] || get(c, "mini", false)
    args["binned"] = (args["binned"] || get(c, "binned", false)) && (args["eos_dir"] == "")
    args["scattering"] = args["scattering"] || get(c, "scattering", false)
    args["use_threads"] = args["use_threads"] || get(c, "use_threads", false)

    return args
end

function main()
    args = parse_commandline()

    eos_dir = if (args["eos_dir"] == "")
        eos_config_path = args["eos_config"]
        if !isfile(eos_config_path)
            error("EoS configuration file not found at: $eos_config_path. Please provide a valid --eos_config or explicitly set --eos_dir.")
        end 
        println("================================================================================")
        println("================== M1DIS.jl + TSO.jl Opacity Tables ============================")
        println("================================================================================")
        @info("Fetching/Computing from composition...")
        eos_c = TOML.parsefile(eos_config_path)

        try
            @import_tumult eos_c["m3d_dir"]
        catch e
            @warn "Could not import Multi3D. Please provide a valid path via --m3d_dir."
        end

        eos_dir = M1DIS.get_or_compute_eos(
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
    else
        args["eos_dir"]
    end

    println("Starting M1DIS Execution...")
    println("Teff: $(args["teff"]) K, logg: $(args["logg"])")
    
    if args["binned"]
        eos_file = joinpath(eos_dir, "eos_T.hdf5")
        eos500_file = joinpath(eos_dir, "eos_T500.hdf5")
        opa_file = joinpath(eos_dir, "binned_opacities_T.hdf5")
        scat_file = ""
    else
        eos_file = MUST.glob("*_eos_*.hdf5", eos_dir)[1]
        eos500_file = MUST.glob("*_eos500_*.hdf5", eos_dir)[1]
        opa_file = MUST.glob("*_opacities_*.hdf5", eos_dir)[1]
        scat_file = MUST.glob("*_sopacities_*.hdf5", eos_dir)[1]
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

    scat_complete = if args["scattering"]
        TSO.reload(TSO.MiniOpacityTable, scat_file)
    else
        nothing
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
        α_MLT = args["alpha_MLT"],
        maxiter = args["maxiter"],
        eos = eos_complete,
        opacity = opa_complete,
        damping = args["damping"],
        τ = 10.0 .^ range(args["tau_min"], args["tau_max"], length=args["n_tau"]),
        use_threads = use_threads,
        feutrier = true,
        scattering_opacity = scat_complete,
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
