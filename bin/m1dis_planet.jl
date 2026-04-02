#!/usr/bin/env julia

# Activate your project environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using M1DIS
using ArgParse
using TSO
using MUST
using TOML

# ==============================================================================
# Constants (CGS Units)
# ==============================================================================
const R_SUN_CM = 6.957e10      # Solar Radius [cm]
const R_JUP_CM = 7.1492e9      # Jupiter Radius [cm]
const M_JUP_G  = 1.898e30      # Jupiter Mass [g]
const AU_CM    = 1.496e13      # Astronomical Unit [cm]
const G_CGS    = 6.6743e-8     # Gravitational Constant [cm^3 g^-1 s^-2]

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

    s = ArgParseSettings(description = "M1DIS.jl Planet Atmosphere Solver")

    @add_arg_table s begin
        "--config", "-c"
            help = "Path to a TOML configuration file. Command line arguments override these values."
            arg_type = String
            default = ""
        "--t_int"
            help = "Internal temperature of the planet (K)"
            arg_type = Float64
            default = convert(Float64, get(c, "t_int", 1200.0))
        "--t_target"
            help = "Target temperature of the planet. This is used as the target flux."
            arg_type = Float64
            default = convert(Float64, get(c, "t_target", -1.0))
        "--a_au"
            help = "Semi-major axis [AU]"
            arg_type = Float64
            default = convert(Float64, get(c, "a_au", 0.047))
        "--r_pl"
            help = "Planetary Radius [R_Jup]"
            arg_type = Float64
            default = convert(Float64, get(c, "r_pl", 1.4))
        "--m_pl"
            help = "Planetary Mass [M_Jup]"
            arg_type = Float64
            default = convert(Float64, get(c, "m_pl", 0.69))
        "--t_star"
            help = "Host Star Effective Temperature [K]"
            arg_type = Float64
            default = convert(Float64, get(c, "t_star", 6065.0))
        "--td"
            help = "Transit Depth [%] (superseded by --r_star if provided)"
            arg_type = Float64
            default = convert(Float64, get(c, "td", 1.5))
        "--r_star"
            help = "Host Star Radius [R_sun] (optional, overrides transit depth)"
            arg_type = Float64
            default = convert(Float64, get(c, "r_star", -1.0))
        "--vmic"
            help = "Microturbulence (km/s)"
            arg_type = Float64
            default = convert(Float64, get(c, "vmic", 0.0))
        "--maxiter"
            help = "Maximum number of iterations"
            arg_type = Int
            default = convert(Int, get(c, "maxiter", 40))
        "--alpha_MLT"
            help = "Mixing length parameter"
            arg_type = Float64
            default = convert(Float64, get(c, "alpha_MLT", 1.5))
        "--pbeta"
            help = "Parameter for turbulent pressure. 0.0 turns off turbulent pressure."
            arg_type = Float64
            default = convert(Float64, get(c, "pbeta", 1.0))
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
        "--eos_dir"
            help = "Path to the directory containing the Equation of State and Opacity table files (required either via CLI or config)"
            default = get(c, "eos_dir", "")
        "--scattering"
            help = "Include scattering opacity separately from true absorption opacity."
            action = :store_true
        "--out_dir"
            help = "Output directory for the saved models"
            default = get(c, "out_dir", "./models")
        "--model_name", "-n"
            help = "Name of the model to save"
            default = get(c, "model_name", "")
        "--mini"
            help = "Ignore source function from the opacity table and recompute it on-the-fly."
            action = :store_true
        "--binned"
            help = "Signal that the opacity table uses binned opacities."
            action = :store_true
        "--tau_min"
            help = "Minimum optical depth to compute."
            arg_type = Float64
            default = convert(Float64, get(c, "tau_min", -7.0))
        "--tau_max"
            help = "Maximum optical depth to compute."
            arg_type = Float64
            default = convert(Float64, get(c, "tau_max", 2.0))
        "--n_tau"
            help = "Number of optical depth points."
            arg_type = Int
            default = convert(Int, get(c, "n_tau", 200))
        "--damping"
            help = "Damping parameter for the temperature updates."
            arg_type = Float64
            default = convert(Float64, get(c, "damping", 0.05))
        "--use_threads"
            help = "Use multi-threading for the RT."
            action = :store_true
        "--F_irradiation"
            help = """
            File with irradiation flux as a function of wavelength (in Å, first column wavelength, second column flux). 
            Will be interpolated to the internal grid of the opacity table.
            """
            arg_type = String
            default = get(c, "F_irradiation", "")
    end

    args = parse_args(s)

    args["mini"] = args["mini"] || get(c, "mini", false)
    args["binned"] = (args["binned"] || get(c, "binned", false)) && (args["eos_dir"] == "")
    args["use_threads"] = args["use_threads"] || get(c, "use_threads", false)
    args["scattering"] = args["scattering"] || get(c, "scattering", false)

    return args
end

function main()
    args = parse_commandline()
    
    eos_dir = if (args["eos_dir"] == "")
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

    # 3. Derived Physical Parameters
    R_pl_cm = args["r_pl"] * R_JUP_CM
    M_pl_g  = args["m_pl"] * M_JUP_G

    g_planet = (G_CGS * M_pl_g) / (R_pl_cm^2)
    logg_planet = log10(g_planet)

    if args["r_star"] > 0.0
        R_star_cm = args["r_star"] * R_SUN_CM
    else
        TD_frac = args["td"] / 100.0
        R_star_cm = R_pl_cm / sqrt(TD_frac)
    end

    d_orbit_cm = args["a_au"] * AU_CM

    dilution_factor = (R_star_cm / d_orbit_cm)^2

    t_internal_planet = args["t_int"]
    T_star_eff = args["t_star"]

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

    F_irr = if args["F_irradiation"] == ""
        nothing
    else
        d = M1DIS.readdlm(args["F_irradiation"], comments=true)
        l = d[:, 1]
        F = d[:, 2]
        m = sortperm(l)

        # the flux is given in units of erg/(s*cm^2*Hz)
        # we need to convert it to erg/(s*cm^2*cm). Since l is in Angstrom, we do
        lcm = l .* TSO.aa_to_cm
        F_cm = F .* MUST.CLight ./ (lcm .^ 2)

        ip = M1DIS.linear_interpolation(l[m], log.(F_cm[m]), extrapolation_bc=M1DIS.Flat())
        exp.(ip.(TSO.wavelength(opa_complete)))
    end

    use_threads = (!args["binned"]) || (args["use_threads"])

    println("================================================================================")
    println("================ M1DIS.jl Planet Atmosphere Solver =============================")
    println("================================================================================")
    println("--- Star & Planet Parameters ---")
    println("Planet Gravity (log g): ", round(logg_planet, digits=3))
    println("Star Temperature (K):   ", T_star_eff)
    println("Star Radius (cm):       ", R_star_cm, " (", round(R_star_cm/R_SUN_CM, digits=2), " R_sun)")
    println("Orbital Distance (cm):  ", d_orbit_cm, " (", args["a_au"], " AU)")
    println("")
    println("Computing atmosphere...")
    M1DIS.activate_timing!()
    M1DIS.start_timing!()
    result = atmosphere(
        T_eff = t_internal_planet,
        target_flux = args["t_target"] < 0.0 ? nothing : M1DIS.σ_SB * args["t_target"]^4,
        logg = logg_planet,
        #v_mac = args["vmic"],
        α_MLT = args["alpha_MLT"],
        pbeta = args["pbeta"],
        maxiter = args["maxiter"],
        eos = eos_complete,
        opacity = opa_complete,
        #solver=:approximate,
        damping = args["damping"],
        τ = 10.0 .^ range(args["tau_min"], args["tau_max"], length=args["n_tau"]),
        use_threads = use_threads,
        feutrier = true,
        T_irradiation = T_star_eff,
        d_irradiation = d_orbit_cm,
        R_irradiation = R_star_cm,
        F_irradiation = F_irr,
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

    model_name, information = if args["model_name"] == ""
        a = args["alpha"]
        z = args["feh"]
        v = args["vmic"]
        t = args["t_star"]
        ti = args["t_int"]
        i = "* T_int [K]\n* $(ti)\n*\n* T_star [K]\n* $(t)\n*\n* [Fe/H]\n* $(z)\n*\n* [alpha/Fe]\n* $(a)\n*\n* vmic [km/s]\n* $(v)\n*"
        "p_tint$(ti)_tstar$(t)_g$(round(logg_planet, digits=2))_z$(z)_a$(a)_vmic$(v)", i
    else
        args["model_name"], nothing
    end
    println("Saving model $(model_name) to $out_dir...")

    # deleting the dir if it exists
    run_dir = joinpath(abspath(out_dir), model_name)
    if isdir(run_dir)
        @info "Output dir already exists. Clearing $(run_dir)."
        rm(run_dir, recursive=true, force=true)
    end
    println("Saving model $(model_name) to $out_dir...")
    save!(
        result[end], model_name; 
        folder = out_dir, 
        vmic = args["vmic"], 
        logg = logg_planet,
        eos500 = eos500_complete, information = information
    )

    # save the iterations also
    if (!isdir(joinpath(out_dir, model_name, "iterations")))
        mkpath(joinpath(out_dir, model_name, "iterations"))
    else
        rm(joinpath(out_dir, model_name, "iterations"), recursive=true)
        mkpath(joinpath(out_dir, model_name, "iterations"))
    end
    for (i, r) in enumerate(result)
        save!(
            r, "iteration_$(i)"; 
            folder = joinpath(out_dir, model_name, "iterations"), 
            vmic = args["vmic"], 
            logg = logg_planet,
            eos500 = eos500_complete
        )
    end
    
    println("M1DIS.jl planet model finished.")
    println("================================================================================")
end

main()
