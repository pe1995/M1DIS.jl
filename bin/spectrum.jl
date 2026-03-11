#!/usr/bin/env julia

#= M3D spectrum synthesis for 1D M1DIS models =#

# Activate your project environment
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MUST 
using TSO
using ArgParse
using TOML

# ==============================================================================
# Argument parsing helpers
# ==============================================================================

struct ChemicalComposition
    abundances ::Dict
end

ArgParse.parse_item(::Type{ChemicalComposition}, x::AbstractString) = begin
    elements_abundances = split(x, ",", keepempty=false)
    elements = [Symbol(split(a, "_", keepempty=false) |> first |> string) for a in elements_abundances]
    abundances = [Float64(Base.parse_input_line(split(a, "_", keepempty=false) |> last) |> eval) for a in elements_abundances]

    ChemicalComposition(Dict(e=>a for (e, a) in zip(elements, abundances)))
end

# ==============================================================================
# Command line + TOML config parsing (same pattern as m1dis_star.jl)
# ==============================================================================

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

    s = ArgParseSettings(description = "M1DIS.jl 1D Spectrum Synthesis via Multi3D")

    @add_arg_table s begin
        "--config", "-c"
            help = "Path to a TOML configuration file. Command line arguments override these values."
            arg_type = String
            default = ""
        "--model_dir", "-m"
            help = """Path to the M1DIS model directory (e.g. \"models/m1dis_model\").
            The code will automatically find the *_m3d.txt file and save results to the HDF5 box."""
            arg_type = String
            default = get(c, "model_dir", "")
        "--eos_config"
            help = "Path to the EoS config file (used for m3d_dir)."
            arg_type = String
            default = get(c, "eos_config", "eos_config.toml")
        "--name", "-n"
            help = "Additional name for the output ('_' added automatically). This is usefull if you want to compute multiple spectra for the same model."
            arg_type = String
            default = get(c, "name", "")
        "--lambda_start", "-s"
            help = "Wavelength window start [Å]."
            arg_type = Float64
            default = convert(Float64, get(c, "lambda_start", 3000.0))
        "--lambda_end", "-e"
            help = "Wavelength window end [Å]."
            arg_type = Float64
            default = convert(Float64, get(c, "lambda_end", 10000.0))
        "--lambda_step", "-d"
            help = "Wavelength step [Å] (used when lambda_n < 0)."
            arg_type = Float64
            default = convert(Float64, get(c, "lambda_step", 0.01))
        "--lambda_n"
            help = "Number of wavelength points (overrides lambda_step if > 0)."
            arg_type = Float64
            default = convert(Float64, get(c, "lambda_n", -1.0))
        "--lambda_log"
            help = "Sample wavelength grid in log units."
            action = :store_true
        "--feh"
            help = "Metallicity [Fe/H]."
            arg_type = Float64
            default = convert(Float64, get(c, "feh", 0.0))
        "--alpha"
            help = "Alpha enhancement [α/Fe]."
            arg_type = Float64
            default = convert(Float64, get(c, "alpha", 0.0))
        "--composition"
            help = """
            Individual element abundances [X/Fe].
            Specify as e.g. "C_0.3,O_0.2,Si_-0.3".
            """
            arg_type = ChemicalComposition
            default = ChemicalComposition(Dict())
        "--vmic"
            help = "Microturbulence velocity [km/s]."
            arg_type = Float64
            default = convert(Float64, get(c, "vmic", 1.0))
        "--linelists"
            help = """List of linelists. Use "all" for the default linelists, "" for none."""
            arg_type = String
            default = get(c, "linelists", "input_multi3d/master_linelists/")
        "--abund"
            help = "Default abundance file."
            arg_type = String
            default = get(c, "abund", "./input_multi3d/abund/abund_magg")
        "--absdat"
            help = "Absdat file."
            arg_type = String
            default = get(c, "absdat", "./input_multi3d/TS_absdat.dat")
        "--absmet"
            help = "Absmet file."
            arg_type = String
            default = get(c, "absmet", "")
        "--atom"
            help = "Model atom file (leave empty for no model atom)."
            arg_type = String
            default = get(c, "atom", "")
        "--atom_lines"
            help = "Lines in the model atom for special treatment (comma-separated indices)."
            arg_type = String
            default = get(c, "atom_lines", "")
        "--NLTE"
            help = "Turn on NLTE."
            action = :store_true
        "--multi_threads", "-t"
            help = "Number of threads for M3D."
            arg_type = Int
            default = convert(Int, get(c, "multi_threads", 20))
        "--nnu"
            help = "Number of frequency splits."
            arg_type = Int
            default = convert(Int, get(c, "nnu", 32))
        "--dims"
            help = "Number of vertical atmosphere splits."
            arg_type = Int
            default = convert(Int, get(c, "dims", 32))
        "--nz"
            help = "Number of vertical grid points for resampling (-1 = 128)."
            arg_type = Int
            default = convert(Int, get(c, "nz", -1))
        "--short_scheme"
            help = "Scheme for short characteristics."
            arg_type = String
            default = get(c, "short_scheme", "radau")
        "--short_ntheta"
            help = "Number of theta angles (short char.)."
            arg_type = Int
            default = convert(Int, get(c, "short_ntheta", 2))
        "--short_nphi"
            help = "Number of phi angles (short char.)."
            arg_type = Int
            default = convert(Int, get(c, "short_nphi", 4))
        "--long_scheme"
            help = "Scheme for long characteristics."
            arg_type = String
            default = get(c, "long_scheme", "lobatto")
        "--long_ntheta"
            help = "Number of theta angles (long char.)."
            arg_type = Int
            default = convert(Int, get(c, "long_ntheta", 4))
        "--long_nphi"
            help = "Number of phi angles (long char.)."
            arg_type = Int
            default = convert(Int, get(c, "long_nphi", 4))
        "--long_mu"
            help = "Custom mu quadrature for long char. (space-separated, e.g. '0.1 0.5 1.0')."
            arg_type = String
            default = get(c, "long_mu", "")
        "--save_chi"
            help = "Save opacities."
            action = :store_true
        "--save_snu"
            help = "Save source function."
            action = :store_true
        "--keep_logs"
            help = "Keep M3D output logs (namelist + log files)."
            action = :store_true
        "--remove"
            help = "Remove M3D raw output after saving to HDF5."
            action = :store_true
    end

    args = parse_args(s)

    # Boolean flags: OR with config values
    args["lambda_log"] = args["lambda_log"] || get(c, "lambda_log", false)
    args["NLTE"] = args["NLTE"] || get(c, "NLTE", false)
    args["save_chi"] = args["save_chi"] || get(c, "save_chi", false)
    args["save_snu"] = args["save_snu"] || get(c, "save_snu", false)
    args["keep_logs"] = args["keep_logs"] || get(c, "keep_logs", false)
    args["remove"] = args["remove"] || get(c, "remove", false)

    # Handle composition from config if not set via CLI
    if isempty(args["composition"].abundances) && haskey(c, "composition") && !isempty(c["composition"])
        args["composition"] = ArgParse.parse_item(ChemicalComposition, c["composition"])
    end

    return args
end

# ==============================================================================
# Main function
# ==============================================================================

function main()
    println("================================================================================")
    println("==================== M1DIS.jl + MUST.jl Spectrum Synthesis =====================")
    println("================================================================================")
    
    args = parse_commandline()

    # ==============================================================================
    # Resolve model directory and find M3D text file
    # ==============================================================================
    model_dir = args["model_dir"]
    if isempty(model_dir)
        error("No model directory provided. Use --model_dir or set model_dir in the config file.")
    end
    model_dir = abspath(model_dir)
    if !isdir(model_dir)
        error("Model directory not found: $model_dir")
    end

    model_dir = model_dir[end] == '/' ?  model_dir[1:end-1] : model_dir
    model_name = basename(model_dir)

    # Find the *_m3d.txt file automatically
    m3d_file = joinpath(model_dir, "$(model_name)_m3d.txt")
    if !isfile(m3d_file)
        candidates = filter(f -> endswith(f, "_m3d.txt"), readdir(model_dir))
        if isempty(candidates)
            error("No *_m3d.txt file found in $model_dir. Expected $(model_name)_m3d.txt")
        end
        m3d_file = joinpath(model_dir, candidates[1])
        @warn "Expected $(model_name)_m3d.txt, using $(candidates[1]) instead."
    end
    m3d_basename = basename(m3d_file)

    # ==============================================================================
    # Load M3D
    # ==============================================================================
    @info "Computing spectra for: $m3d_basename"
    eos_config_path = args["eos_config"]
    if !isabspath(eos_config_path)
        eos_config_path = joinpath(@__DIR__, eos_config_path)
    end
    if !isfile(eos_config_path)
        error("EoS configuration file not found at: $eos_config_path. Please provide a valid --eos_config.")
    end
    eos_c = TOML.parsefile(eos_config_path)
    m3d_dir = get(eos_c, "m3d_dir", "")
    if isempty(m3d_dir)
        error("m3d_dir not set in $(eos_config_path). Please specify the Multi3D installation path.")
    end
    MUST.@import_tumult m3d_dir

    # ==============================================================================
    # Wavelength grid
    # ==============================================================================
    λs = args["lambda_log"] ? log(args["lambda_start"]) : args["lambda_start"]
    λe = args["lambda_log"] ? log(args["lambda_end"]) : args["lambda_end"]
    Δλ, nλ = if args["lambda_n"] < 0.0
        Δλ = args["lambda_log"] ? log(args["lambda_step"]) : args["lambda_step"]
        nλ = (λe - λs) / Δλ
        Δλ, nλ
    else
        nλ = args["lambda_n"]
        Δλ = (λe - λs) / nλ
        Δλ, nλ
    end
    window = MUST.@sprintf "lam_%i-%i" λs λe
    window_nice = MUST.@sprintf "%iÅ - %iÅ" λs λe
    extension = length(args["name"]) > 0 ? "_"*args["name"] : ""
    prefix = length(args["name"]) > 0 ? args["name"]*"_" : ""

    @info "Spectral window: $(window_nice)."

    # ==============================================================================
    # Linelists
    # ==============================================================================
    linelists = args["linelists"]

    # ==============================================================================
    # Chemical composition
    # ==============================================================================
    FeH = args["feh"]
    α = args["alpha"]
    composition = args["composition"]
    abund_file = MUST.abund_abundances(;
        α=α,
        Dict(k=>(lowercase(String(k)) in ["he", "li"]) ? v + FeH : v for (k,v) in composition.abundances)...,
        default=args["abund"]
    )

    cs = join(["[$(k)/Fe]=$(v)" for (k, v) in composition.abundances], ",")
    cstring = length(cs) > 0 ? "[Fe/H]=$(FeH),[α/Fe]=$(α),$(cs)" : "[Fe/H]=$(FeH),[α/Fe]=$(α)"
    @info "Chemical composition: $(cstring), saved at $(abund_file)."

    # ==============================================================================
    # Model atom
    # ==============================================================================
    atom_params, line_mask, spectrum_params = if (length(args["atom"]) > 0) && (λs < 0)
        @info "Leaving `spectrum_params` empty."
        ma = args["atom"]

        d_atom, d_line = if length(args["atom_lines"]) == 0
            Dict(:atom_file=>ma, :use_atom_abnd=>false), Dict()
        else
            input_lines = parse.(Int, split(args["atom_lines"], ',', keepempty=false))
            @info "Computing contribution functions for $(input_lines)"
            d = Dict(
                :atom_file=>ma, :use_atom_abnd=>false,
                :cbbesc=>1, :cbfesc=>1,
                :cbbhsc=>1, :cbfhsc=>1,
                :level_partf=>false,
                :n_in_lines=>length(input_lines),
                :n_cntrbf=>length(input_lines),
            )
            dl = Dict(
                :in_lines => input_lines,
                :cntrbf_lines => input_lines
            )
            d, dl
        end
        d_atom, d_line, Dict()

    elseif length(args["atom"]) > 0
        ma = args["atom"]
        @info "Using model atom ($(ma))."

        d = Dict(:atom_file=>ma, :use_atom_abnd=>false)
        d, Dict(), Dict(:daa=>Δλ, :aa_blue=>λs, :aa_red=>λe, :in_log=>args["lambda_log"])
    else
        @info "No model atom specified."
        Dict(), Dict(), Dict(:daa=>Δλ, :aa_blue=>λs, :aa_red=>λe, :in_log=>args["lambda_log"])
    end

    # ==============================================================================
    # Angular quadrature
    # ==============================================================================
    angle_params = if length(args["long_mu"]) > 0
        Dict(
            :long_nphi=>args["long_nphi"],
            :long_scheme=>"custom",
            :custom_mu=>"''"*args["long_mu"]*"''",
        )
    else
        Dict(
            :long_nphi=>args["long_nphi"],
            :long_ntheta=>args["long_ntheta"],
            :long_scheme=>args["long_scheme"],
        )
    end

    # ==============================================================================
    # General namelist adjustments
    # ==============================================================================
    nz = args["nz"] == -1 ? 128 : args["nz"]
    spectrum_namelist = Dict(
        :model_folder=>model_dir,
        :linelist=>nothing,
        :absmet=>length(args["absmet"])==0 ? nothing : args["absmet"],
        :linelist_params=>(:linelist_folder=>linelists,),
        :atom_params=>(atom_params...,),
        :line_mask=>(line_mask...,),
        :spectrum_params=>(spectrum_params...,),
        :atmos_params=>(
            :dims=>args["dims"],
            :atmos_format=>"text",
            :use_rho=>true,
            :use_ne=>false,
            :FeH=>FeH,
            :amr=>false,
            :nz=>nz
        ),
        :m3d_params=>(
            :short_scheme=>args["short_scheme"],
            :n_nu=>args["nnu"],
            :decouple_continuum=>true,
            :save_chi=>args["save_chi"],
            :save_snu=>args["save_snu"],
            :short_ntheta=>args["short_ntheta"],
            :short_nphi=>args["short_nphi"],
            angle_params...
        ),
        :composition_params=>(
            :absdat_file=>args["absdat"],
            :abund_file=>abund_file,
        )
    )

    if args["vmic"] > 0
        spectrum_namelist[:atmos_params] = (:vmic=>args["vmic"], spectrum_namelist[:atmos_params]...)
    end

    # ==============================================================================
    # Run Multi3D
    # ==============================================================================
    m3dis_kwargs = Dict(:threads=>args["multi_threads"])

    result = try
        r = MUST.spectrum(
            m3d_basename;
            name=model_name*"_"*window*extension,
            NLTE=args["NLTE"],
            slurm=false,
            namelist_kwargs=spectrum_namelist,
            m3dis_kwargs=m3dis_kwargs,
            twostep=false,
            cleanup=!args["keep_logs"]
        )
        @info "✅ Spectrum synthis completed."

        r
    catch e
        @warn "❌ Spectrum synthis failed."
        error(e)
        nothing
    end

    # =========================================================================
    # Build unique spectrum identifier from composition + atom
    # =========================================================================
    spec_id_parts = String[]
    push!(spec_id_parts, "feh_$(round(FeH, digits=2))")
    if α != 0.0
        push!(spec_id_parts, "alpha_$(round(α, digits=2))")
    end
    for (k, v) in sort(collect(composition.abundances), by=first)
        push!(spec_id_parts, "$(k)_$(round(v, digits=2))")
    end
    if args["NLTE"] && length(args["atom"]) > 0
        atom_base = basename(args["atom"])
        push!(spec_id_parts, "NLTE_$(atom_base)")
    end
    spec_id = join(spec_id_parts, ":")
    spec_name = (length(prefix)==0) ? spec_id : args["name"] #* spec_id

    name_cat(n) = Symbol(spec_name * "_$(n)")

    # =========================================================================
    # Save spectrum to HDF5 box in model directory
    # =========================================================================
    box_file = joinpath(model_dir, "$(model_name).hdf5")
    if !isfile(box_file)
        @warn "No HDF5 box found at $(box_file). Creating a basic box from M3D output."
        x = Base.convert.(Float32, MUST.pyconvert(Array, result.run.atmos.xx) .* 1e8)
        y = Base.convert.(Float32, MUST.pyconvert(Array, result.run.atmos.yy) .* 1e8)
        z = Base.convert.(Float32, MUST.pyconvert(Array, result.run.atmos.zz) .* 1e8)
        set_size(v) = reshape(v, 1, 1, length(v))
        data = Dict{Symbol, Any}(
            :T => set_size(MUST.pyconvert(Array, result.run.atmos.temp)),
            :d => set_size(MUST.pyconvert(Array, result.run.atmos.rho)),
        )
        xx, yy, zz = MUST.meshgrid(x, y, z)
        b2 = MUST.Box(xx, yy, zz, data, MUST.AtmosphericParameters())
        MUST.save(b2, folder=model_dir, name=model_name)
    end

    b2 = MUST.Box(model_name, folder=model_dir)

    ii = MUST.pyconvert(Array, result.run.ie)
    cc = MUST.pyconvert(Array, result.run.ie_cnt)
    lam = MUST.pyconvert(Array, result.run.lam)
    wts = MUST.pyconvert(Array, result.run.wts)
    mu_vec = MUST.pyconvert(Array, result.run.vec)

    b2.data = Dict{Symbol, Any}()
    b2.data[name_cat("weights")] = wts
    b2.data[name_cat("wavelength")] = lam
    b2.data[name_cat("mu")] = mu_vec
    b2.data[name_cat("composition")] = cstring

    b2.data[name_cat("intensity")] = reshape(ii, size(ii, 1), 1, 1, size(ii, 2))
    b2.data[name_cat("continuum")] = reshape(cc, size(cc, 1), 1, 1, size(cc, 2))
    b2.data[name_cat("meanFlux")] = MUST.mean_integrated_flux(b2, spec_name, norm=false) |> last
    b2.data[name_cat("meanFluxNorm")] = MUST.mean_integrated_flux(b2, spec_name, norm=true) |> last
    b2.data[name_cat("meanIntensity")] = MUST.mean_intensity(b2, spec_name, norm=false) |> last
    b2.data[name_cat("meanIntensityNorm")] = MUST.mean_intensity(b2, spec_name, norm=true) |> last

    ii_lte = if args["NLTE"]
        ii_lte = MUST.pyconvert(Array, result.run.ie_lte)
        b2.data[name_cat("intensityLTE")] = reshape(ii_lte, size(ii_lte, 1), 1, 1, size(ii_lte, 2))
        b2.data[name_cat("meanFluxLTE")] = MUST.mean_integrated_flux(b2, spec_name, norm=false, intensity="intensityLTE") |> last
        b2.data[name_cat("meanFluxNormLTE")] = MUST.mean_integrated_flux(b2, spec_name, norm=true, intensity="intensityLTE") |> last
        b2.data[name_cat("meanIntensityLTE")] = MUST.mean_intensity(b2, spec_name, norm=false, intensity="intensityLTE") |> last
        b2.data[name_cat("meanIntensityNormLTE")] = MUST.mean_intensity(b2, spec_name, norm=true, intensity="intensityLTE") |> last
        ii_lte
    else
        nothing
    end

    MUST.save(b2, folder=model_dir, name=model_name, mode="r+")
    @info "Spectrum saved to $(model_name) under the tag '$(spec_name)'."

    # =========================================================================
    # Save flux and intensity as text files
    # =========================================================================
    spectra_dir = joinpath(model_dir, "spectra")
    if !isdir(spectra_dir)
        mkpath(spectra_dir)
    end

    teff_val = b2.parameter.teff
    logg_val = b2.parameter.logg
    mu_z = mu_vec[:, 3]
    nlte_line = args["NLTE"] ? "# NLTE atom: $(args["atom"])" : "# LTE"

    # Flux
    flux_data = b2.data[name_cat("meanFlux")]
    flux_data_lte = args["NLTE"] ? b2.data[name_cat("meanFluxLTE")] : nothing
    flux_file = joinpath(spectra_dir, "flux$(extension).txt")
    open(flux_file, "w") do f
        println(f, "# M1DIS Spectrum: Flux")
        println(f, "# Teff = $(teff_val) K")
        println(f, "# logg = $(logg_val)")
        println(f, "# Composition: $(cstring)")
        println(f, "# Type: Flux (disk-integrated)")
        println(f, nlte_line)
        println(f, "# Column 1: Wavelength [Å]")
        println(f, "# Column 2: Flux (LTE) [erg/s/cm^2/Å]")
        args["NLTE"] && println(f, "# Column 3: Flux (NLTE) [erg/s/cm^2/Å]")
        for i in eachindex(lam)
            if args["NLTE"]
                s = MUST.@sprintf("%10.5f %10.5E %10.5E", lam[i], flux_data_lte[i], flux_data[i])
                println(f, s)
            else
                s = MUST.@sprintf("%10.5f %10.5E", lam[i], flux_data[i])
                println(f, s)
            end
        end
    end

    # Normalized flux
    flux_norm_data = b2.data[name_cat("meanFluxNorm")]
    flux_norm_data_lte = args["NLTE"] ? b2.data[name_cat("meanFluxNormLTE")] : nothing
    flux_norm_file = joinpath(spectra_dir, "flux_norm$(extension).txt")
    open(flux_norm_file, "w") do f
        println(f, "# M1DIS Spectrum: Normalized Flux")
        println(f, "# Teff = $(teff_val) K")
        println(f, "# logg = $(logg_val)")
        println(f, "# Composition: $(cstring)")
        println(f, "# Type: Flux (disk-integrated, continuum-normalized)")
        println(f, nlte_line)
        println(f, "# Column 1: Wavelength [Å]")
        println(f, "# Column 2: Normalized Flux (LTE)")
        args["NLTE"] && println(f, "# Column 3: Normalized Flux (NLTE)")
        for i in eachindex(lam)
            if args["NLTE"]
                s = MUST.@sprintf("%10.5f %10.5E %10.5E", lam[i], flux_norm_data_lte[i], flux_norm_data[i])
                println(f, s)
            else
                s = MUST.@sprintf("%10.5f %10.5E", lam[i], flux_norm_data[i])
                println(f, s)
            end
        end
    end

    # Intensity per mu angle
    ii_norm = b2.data[name_cat("meanIntensityNorm")]
    ii_norm_lte = args["NLTE"] ? b2.data[name_cat("meanIntensityNormLTE")] : nothing
    for a in axes(ii, 2)
        mu_val = round(mu_z[a], digits=4)
        int_file = joinpath(spectra_dir, "intensity_mu$(mu_val)$(extension).txt")
        open(int_file, "w") do f
            println(f, "# M1DIS Spectrum: Intensity")
            println(f, "# Teff = $(teff_val) K")
            println(f, "# logg = $(logg_val)")
            println(f, "# Composition: $(cstring)")
            println(f, "# Type: Specific Intensity at mu = $(mu_val)")
            println(f, nlte_line)
            println(f, "# Column 1: Wavelength [Å]")
            println(f, "# Column 2: Intensity (LTE) [erg/s/cm^2/Å/sr]")
            args["NLTE"] && println(f, "# Column 3: Intensity (NLTE) [erg/s/cm^2/Å/sr]")
            for i in axes(ii, 1)
                if args["NLTE"]
                    s = MUST.@sprintf("%10.5f %10.5E %10.5E", lam[i], ii_lte[i, a], ii[i, a])
                    println(f, s)
                else
                    s = MUST.@sprintf("%10.5f %10.5E", lam[i], ii[i, a])
                    println(f, s)
                end
            end
        end
        int_file = joinpath(spectra_dir, "intensity_norm_mu$(mu_val)$(extension).txt")
        open(int_file, "w") do f
            println(f, "# M1DIS Spectrum: Intensity")
            println(f, "# Teff = $(teff_val) K")
            println(f, "# logg = $(logg_val)")
            println(f, "# Composition: $(cstring)")
            println(f, "# Type: Specific Intensity at mu = $(mu_val)")
            println(f, nlte_line)
            println(f, "# Column 1: Wavelength [Å]")
            println(f, "# Column 2: Normalized intensity (LTE)")
            args["NLTE"] && println(f, "# Column 3: Normalized intensity (NLTE)")
            for i in axes(ii, 1)
                if args["NLTE"]
                    s = MUST.@sprintf("%10.5f %10.5E %10.5E", lam[i], ii_norm_lte[i, a], ii_norm[i, a])
                    println(f, s)
                else
                    s = MUST.@sprintf("%10.5f %10.5E", lam[i], ii_norm[i, a])
                    println(f, s)
                end
            end
        end
    end

    @info "Spectrum also saved to $(spectra_dir)."

    # ==============================================================================
    # Clean up M3D raw output if requested
    # ==============================================================================
    if args["remove"]
        rawp = @in_m3dis("data/$(m3d_basename)_$(model_name)_$(window)$(extension)")
        if isdir(rawp)
            @info "Removing M3D raw output at $(rawp)."
            rm(rawp, force=true, recursive=true)
        end
    end

    @info "Spectrum synthesis finished."
end

main()