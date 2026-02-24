#!/usr/bin/env julia

# Activate your project environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using M1DIS
using ArgParse
using TSO
using MUST

# ==============================================================================
# Constants (CGS Units)
# ==============================================================================
const R_SUN_CM = 6.957e10      # Solar Radius [cm]
const R_JUP_CM = 7.1492e9      # Jupiter Radius [cm]
const M_JUP_G  = 1.898e30      # Jupiter Mass [g]
const AU_CM    = 1.496e13      # Astronomical Unit [cm]
const G_CGS    = 6.6743e-8     # Gravitational Constant [cm^3 g^-1 s^-2]

function parse_commandline()
    s = ArgParseSettings(description = "M1DIS.jl Planet Atmosphere Solver")

    @add_arg_table s begin
        "--t_int"
            help = "Internal temperature of the planet (K)"
            arg_type = Float64
            default = 1200.0
        "--a_au"
            help = "Semi-major axis [AU]"
            arg_type = Float64
            default = 0.047
        "--r_pl"
            help = "Planetary Radius [R_Jup]"
            arg_type = Float64
            default = 1.4
        "--m_pl"
            help = "Planetary Mass [M_Jup]"
            arg_type = Float64
            default = 0.69
        "--t_star"
            help = "Host Star Effective Temperature [K]"
            arg_type = Float64
            default = 6065.0
        "--td"
            help = "Transit Depth [%] (superseded by --r_star if provided)"
            arg_type = Float64
            default = 1.5
        "--r_star"
            help = "Host Star Radius [R_sun] (optional, overrides transit depth)"
            arg_type = Float64
            default = -1.0
        "--vmic"
            help = "Microturbulence (km/s)"
            arg_type = Float64
            default = 0.0
        "--maxiter"
            help = "Maximum number of iterations"
            arg_type = Int
            default = 40
        "--alpha"
            help = "Mixing length parameter"
            arg_type = Float64
            default = 1.5
        "--eos_dir"
            help = "Path to the directory containing the Equation of State and Opacity table files (required)"
            required = true
        "--out_dir"
            help = "Output directory for the saved models"
            default = ""
        "--model_name"
            help = "Name of the model to save"
            default = "m1dis_planet_model"
        "--mini"
            help = "Ignore source function from the opacity table and recompute it on-the-fly."
            action = :store_true
        "--binned"
            help = "Signal that the opacity table uses binned opacities."
            action = :store_true
        "--tau_min"
            help = "Minimum optical depth to compute."
            arg_type = Float64
            default = -7.0
        "--tau_max"
            help = "Maximum optical depth to compute."
            arg_type = Float64
            default = 2.0
        "--n_tau"
            help = "Number of optical depth points."
            arg_type = Int
            default = 200
        "--damping"
            help = "Damping parameter for the temperature updates."
            arg_type = Float64
            default = 0.05
        "--use_threads"
            help = "Use multi-threading for the RT."
            action = :store_true
        "--F_irradiation"
            help = """
            File with irradiation flux as a function of wavelength (in Å, first column wavelength, second column flux). 
            Will be interpolated to the internal grid of the opacity table.
            """
            arg_type = String
            default = ""
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
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
        eos_file = joinpath(args["eos_dir"], "eos_T.hdf5")
        eos500_file = joinpath(args["eos_dir"], "eos_T500.hdf5")
        opa_file = joinpath(args["eos_dir"], "binned_opacities_T.hdf5")
    else
        eos_file = MUST.glob("*_eos_*.hdf5", args["eos_dir"])[1]
        eos500_file = MUST.glob("*_eos500_*.hdf5", args["eos_dir"])[1]
        opa_file = MUST.glob("*_opacities_*.hdf5", args["eos_dir"])[1]
    end

    println("Loading Equation of State from: $eos_file")
    eos_complete = TSO.reload(eos_file)
    
    println("Loading Equation of State (500nm) from: $eos500_file")
    eos500_complete = TSO.reload(eos500_file)

    println("Loading Opacity from: $opa_file")
    opa_complete = if args["mini"]
        TSO.reload(TSO.MiniOpacityTable, opa_file)
    else
        TSO.ExtendedOpacity(opa=TSO.reload(opa_file, mmap=!args["binned"]), binned=args["binned"])
    end

    F_irr = if args["F_irradiation"] == ""
        nothing
    else
        d = M1DIS.readdlm(args["F_irradiation"])
        l = d[:, 1]
        F = d[:, 2]
        m = sortperm(l)
        ip = M1DIS.linear_interpolation(l[m], F[m])
        ip.(TSO.wavelength(opa_complete))
    end

    use_threads = (!args["binned"]) || (args["use_threads"])
    @show use_threads

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
        logg = logg_planet,
        v_mic = args["vmic"],
        α_MLT = args["alpha"],
        maxiter = args["maxiter"],
        eos = eos_complete,
        opacity = opa_complete,
        damping = args["damping"],
        τ = 10.0 .^ range(args["tau_min"], args["tau_max"], length=args["n_tau"]),
        use_threads = use_threads,
        feutrier = true,
        T_irradiation = T_star_eff,
        d_irradiation = d_orbit_cm,
        R_irradiation = R_star_cm,
        F_irradiation = F_irr
    )
    M1DIS.end_timing!()
    
    out_dir = if args["out_dir"] == ""
        joinpath(@__DIR__, "../models")
    else
        args["out_dir"]
    end
    if !isdir(out_dir)
        println("Creating output directory: $out_dir")
        mkpath(out_dir)
    end

    println("Saving model $(args["model_name"]) to $out_dir...")
    save!(result[end], args["model_name"]; 
          folder = out_dir, 
          vmic = args["vmic"], 
          logg = logg_planet,
          eos500 = eos500_complete)
    
    println("M1DIS.jl planet model finished.")
    println("================================================================================")
end

main()
