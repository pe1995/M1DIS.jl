#!/usr/bin/env julia

# Activate your project environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using M1DIS
using ArgParse
using TSO
using MUST

function parse_commandline()
    s = ArgParseSettings(description = "M1DIS.jl Atmosphere Solver")

    @add_arg_table s begin
        "--teff"
            help = "Effective temperature (K)"
            arg_type = Float64
            default = 5777.0
        "--logg"
            help = "Surface gravity"
            arg_type = Float64
            default = 4.44
        "--vmic"
            help = "Microturbulence (km/s)"
            arg_type = Float64
            default = 0.0
        "--maxiter"
            help = "Maximum number of iterations"
            arg_type = Int
            default = 20
        "--alpha"
            help = "Mixing length parameter"
            arg_type = Float64
            default = 1.5
        "--eos_dir"
            help = "Path to the directory containing the Equation of State and Opacity table files (required)"
            required = true
        "--out_dir"
            help = "Output directory for the saved models"
            default = "./models"
        "--model_name"
            help = "Name of the model to save"
            default = "m1dis_model"
        "--mini"
            help = "Ignore source function from the opacity table and recompute it on-the-fly."
            action = :store_true
        "--binned"
            help = "Signal that the opacity table uses binned opacities."
            action = :store_true
        "--tau_min"
            help = "Minimum optical depth to compute."
            arg_type = Float64
            default = -6.0
        "--tau_max"
            help = "Maximum optical depth to compute."
            arg_type = Float64
            default = 2.0
        "--n_tau"
            help = "Number of optical depth points."
            arg_type = Int
            default = 100
        "--damping"
            help = "Damping parameter for the temperature updates."
            arg_type = Float64
            default = 0.1
        "--use_threads"
            help = "Use multi-threading for the RT."
            action = :store_true
        #"--T_irradiation"
        #    help = "Temperature of the irradiation source."
        #    arg_type = Float64
        #    default = -1.0
        #"--d_irradiation"
        #    help = "Distance to the irradiation source."
        #    arg_type = Float64
        #    default = -1.0
        #"--R_irradiation"
        #    help = "Radius of the irradiation source."
        #    arg_type = Float64
        #    default = -1.0
        #"--F_irradiation"
        #    help = """
        #    File with irradiation flux as a function of wavelength (in Å, first column wavelength, second column flux). 
        #    Will be interpolated to the internal grid of the opacity table.
        #    """
        #    arg_type = String
        #    default = ""
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
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

    println("Loading Equation of State from: $eos_file")
    eos_complete = TSO.reload(eos_file)
    
    println("Loading Equation of State (500nm) from: $eos500_file")
    eos500_complete = TSO.reload(eos500_file)

    println("Loading Opacity from: $opa_file")
    opa_complete = if args["mini"]
        TSO.reload(TSO.MiniOpacityTable, opa_file)
    else
        TSO.ExtendedOpacity(TSO.reload(opa_file, mmap=!args["binned"]), binned=args["binned"])
    end

    #T_irr = args["T_irradiation"] < 0.0 ? nothing : args["T_irradiation"]
    #d_irr = args["d_irradiation"] < 0.0 ? nothing : args["d_irradiation"]
    #R_irr = args["R_irradiation"] < 0.0 ? nothing : args["R_irradiation"]
    #F_irr = if args["F_irradiation"] == ""
    #    nothing
    #else
    #    d = M1DIS.readdlm(args["F_irradiation"])
    #    l = d[:, 1]
    #    F = d[:, 2]
    #    m = sortperm(l)
    #    ip = M1DIS.linear_interpolation(l[m], F[m])
    #    ip.(TSO.wavelength(opa_complete))
    #end

    println("================================================================================")
    println("===================== M1DIS.jl Atmosphere Solver ===============================")
    println("================================================================================")
    println("Computing atmosphere...")
    M1DIS.start_timing!()
    result = atmosphere(
        T_eff = args["teff"],
        logg = args["logg"],
        v_mic = args["vmic"],
        α_MLT = args["alpha"],
        maxiter = args["maxiter"],
        eos = eos_complete,
        opacity = opa_complete,
        damping = args["damping"],
        τ = 10.0 .^ range(args["tau_min"], args["tau_max"], length=args["n_tau"]),
        use_threads = (!args["binned"]) || (args["use_threads"]),
        feutrier = true,
        #T_irradiation = T_irr,
        #d_irradiation = d_irr,
        #R_irradiation = R_irr,
        #F_irradiation = F_irr
    )
    M1DIS.end_timing!()
    
    out_dir = args["out_dir"]
    if !isdir(out_dir)
        println("Creating output directory: $out_dir")
        mkpath(out_dir)
    end

    println("Saving model $(args["model_name"]) to $out_dir...")
    save!(result[end], args["model_name"]; 
          folder = out_dir, 
          vmic = args["vmic"], 
          logg = args["logg"],
          eos500 = eos500_complete)
    
    println("M1DIS.jl finished.")
    println("================================================================================")
end

main()
