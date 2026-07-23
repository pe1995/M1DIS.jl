# ============================================================================
# RT components
# ============================================================================

function generate_mu_grid(n_points::Integer)
    x, w = gausslegendre(n_points)
    return @. x / 2 + 0.5, @. w / 2
end

# ============================================================================
# Opacity computations for Feutrier solvers
# ============================================================================

function compute_opacities(eos, opa::TSO.ExtendedOpacity, T, ρ)
    chi = zeros(Float64, length(opa.opa.λ), length(T))
    chi_ref = zeros(Float64, length(T))
    B = zeros(Float64, length(opa.opa.λ), length(T))
    dBdT = zeros(Float64, length(opa.opa.λ), length(T))
    dchidT = zeros(Float64, length(opa.opa.λ), length(T))

    compute_opacities!(chi, chi_ref, B, dBdT, dchidT, eos, opa, T, ρ)

    return chi, chi_ref, B, dBdT, dchidT
end

function compute_opacities!(chi, chi_ref, B, dBdT, dchidT, eos, opa::TSO.ExtendedOpacity, T, ρ)
    lnrho = log.(ρ)
    lnt = log.(T)

    # If dK_dT exists, we could use it. Otherwise, sample dS_dT and fallback to 0.0 for dchidT
    if haskey(opa.extensions, :dκ_dT)
         sample!((chi, B, dBdT, dchidT, chi_ref), eos, opa, (:κ, :src, :dS_dT, :dκ_dT, :lnRoss), lnrho, lnt)
         dchidT .= dchidT .* ρ'
    else
         sample!((chi, B, dBdT, chi_ref), eos, opa, (:κ, :src, :dS_dT, :lnRoss), lnrho, lnt)
         dchidT .= 0.0
    end
    chi_ref .= exp.(chi_ref) .* ρ
    
    return nothing
end

# mini opacity tables require to be called with the chunked version
compute_opacities(eos, opa::TSO.MiniOpacityTable, T, ρ) = compute_opacities_chunked(eos, opa, T, ρ)
compute_opacities!(chi, chi_ref, B, dBdT, dchidT, eos, opa::TSO.MiniOpacityTable, T, ρ) = _compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa, T, ρ, opa.opacity.λ)

# ============================================================================
# Chunked opacity computation for large tables
# ============================================================================

compute_opacities_chunked(eos, opa::TSO.ExtendedOpacity, T, ρ; kwargs...) = compute_opacities_chunked(eos, opa, T, ρ, opa.opa.λ; kwargs...)
compute_opacities_chunked(eos, opa::TSO.MiniOpacityTable, T, ρ; kwargs...) = compute_opacities_chunked(eos, opa, T, ρ, opa.opacity.λ; kwargs...)

function compute_opacities_chunked(eos, opa, T, ρ, λ; opacity_only=false)
    chi = zeros(Float64, length(λ), length(T))
    chi_ref = opacity_only ? nothing : zeros(Float64, length(T))
    B = opacity_only ? nothing : zeros(Float64, length(λ), length(T))
    dBdT = opacity_only ? nothing : zeros(Float64, length(λ), length(T))
    dchidT = opacity_only ? nothing : zeros(Float64, length(λ), length(T))

    _compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa, T, ρ, λ) 

    return chi, chi_ref, B, dBdT, dchidT
end

compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa::TSO.ExtendedOpacity, T, ρ) = _compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa, T, ρ, opa.opa.λ)
compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa::TSO.MiniOpacityTable, T, ρ) = _compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa, T, ρ, opa.opacity.λ)

# ============================================================================
# Chunked core computations
# ============================================================================

_is_binned(opa::TSO.ExtendedOpacity) = opa.binned
_is_binned(opa::TSO.MiniOpacityTable) = false

function _compute_opacities_chunked!(chi, chi_ref, B, dBdT, dchidT, eos, opa, T, ρ, λ)
    lnrho = log.(ρ)
    lnt = log.(T)
    
    grid_T = TSO.EnergyAxis(eos.eos).values
    grid_Rho = TSO.DensityAxis(eos.eos).values
    n_shells = length(lnrho)
    
    coefs_Rho, coefs_T = TSO.weights(eos, lnrho, lnt)
    λ_indices = eachindex(λ)
    chunk_size = cld(length(λ_indices), Threads.nthreads())
    
    ρ_eff = _is_binned(opa) ? ones(eltype(ρ), length(ρ)) : ρ
    
    tasks = map(Iterators.partition(λ_indices, chunk_size)) do range
        Threads.@spawn _compute_opacities_chunk!(range, chi, B, dBdT, dchidT, opa, coefs_T, coefs_Rho, ρ_eff, T, grid_T)
    end
    wait.(tasks)
    
    if !isnothing(chi_ref)
        sample!((chi_ref,), eos, (:lnRoss,), lnrho, lnt)
        chi_ref .= exp.(chi_ref) .* ρ
    end
    
    return nothing
end

@inline _compute_opacities_chunk!(range, chi, B, dBdT, dchidT, opa::TSO.MiniOpacityTable, coefs_T, coefs_Rho, ρ, T, grid_T) = begin
    _inner_compute_chunk!(range, chi, B, dBdT, dchidT, opa.opacity.κ, opa.opacity.λ, opa.weights, coefs_T, coefs_Rho, ρ, T, grid_T)
end

@inline _compute_opacities_chunk!(range, chi, B, dBdT, dchidT, opa::TSO.ExtendedOpacity, coefs_T, coefs_Rho, ρ_eff, T, grid_T) = begin
    _inner_compute_chunk_extended!(range, chi, B, dBdT, dchidT, opa.opa.κ, opa.opa.src, opa.weights, coefs_T, coefs_Rho, ρ_eff, T, grid_T)
end

# ExtendedOpacity tables can be either binned or unbinned.
# If unbinned, weights are set according to midpoint integration. Rho is multiplied to the opacity.
# If binned, weights are set to 1. Rho is not multiplied to the opacity, as this happened during the binning. In this case rho is set to 1.
function _inner_compute_chunk_extended!(range, chi, B, dBdT, dchidT, kappa_data, src_data, weights, coefs_T, coefs_Rho, ρ, T, grid_T)
    n_shells = length(coefs_T)
    nT = length(grid_T)

    @inbounds for i in range
        # w_i is 1 if the table is binned
        w_i = weights[i]
        for j in 1:n_shells
            ct = coefs_T[j]
            cr = coefs_Rho[j]
            
            it, ir = ct.idx, cr.idx
            
            w00 = ct.w_low  * cr.w_low
            w10 = ct.w_high * cr.w_low
            w01 = ct.w_low  * cr.w_high
            w11 = ct.w_high * cr.w_high
            
            val_k = w00 * log(kappa_data[it,   ir,   i]) +
                    w10 * log(kappa_data[it+1, ir,   i]) +
                    w01 * log(kappa_data[it,   ir+1, i]) +
                    w11 * log(kappa_data[it+1, ir+1, i])
            chi[i, j] = exp(val_k) * ρ[j]

            val_s = w00 * log(src_data[it,   ir,   i]) +
                    w10 * log(src_data[it+1, ir,   i]) +
                    w01 * log(src_data[it,   ir+1, i]) +
                    w11 * log(src_data[it+1, ir+1, i])
            B[i, j] = exp(val_s) * w_i

            dS, dT_diff = if (it > 1) && (it < nT)
                log(src_data[it+1, ir, i]) - log(src_data[it-1, ir, i]), 
                grid_T[it+1] - grid_T[it-1]
            elseif it == 1
                log(src_data[it+1, ir, i]) - log(src_data[it, ir, i]), 
                grid_T[it+1] - grid_T[it]
            else
                log(src_data[it, ir, i]) - log(src_data[it-1, ir, i]), 
                grid_T[it] - grid_T[it-1]
            end
            
            dBdT[i, j] = exp(val_s - grid_T[it]) * (dS / dT_diff) * w_i
            
            dChi = if (it > 1) && (it < nT)
                log(kappa_data[it+1, ir, i]) - log(kappa_data[it-1, ir, i])
            elseif it == 1
                log(kappa_data[it+1, ir, i]) - log(kappa_data[it, ir, i])
            else
                log(kappa_data[it, ir, i]) - log(kappa_data[it-1, ir, i])
            end
            
            dchidT[i, j] = exp(val_k - grid_T[it]) * (dChi / dT_diff) * ρ[j]
        end
    end
end
function _inner_compute_chunk_extended!(range, chi, B::Nothing, dBdT::Nothing, dchidT::Nothing, kappa_data, src_data, weights, coefs_T, coefs_Rho, ρ, T, grid_T)
    n_shells = length(coefs_T)
    nT = length(grid_T)

    @inbounds for i in range
        # w_i is 1 if the table is binned
        w_i = weights[i]
        for j in 1:n_shells
            ct = coefs_T[j]
            cr = coefs_Rho[j]
            
            it, ir = ct.idx, cr.idx
            
            w00 = ct.w_low  * cr.w_low
            w10 = ct.w_high * cr.w_low
            w01 = ct.w_low  * cr.w_high
            w11 = ct.w_high * cr.w_high
            
            val_k = w00 * log(kappa_data[it,   ir,   i]) +
                    w10 * log(kappa_data[it+1, ir,   i]) +
                    w01 * log(kappa_data[it,   ir+1, i]) +
                    w11 * log(kappa_data[it+1, ir+1, i])
            chi[i, j] = exp(val_k) * ρ[j]
        end
    end
end

# MiniOpacityTable always are unbinned, as they dont have the source function stored
# On-thy-fly Planck function only works for monochromatic opacities. This means that weights are always multiplied.
function _inner_compute_chunk!(range, chi, B, dBdT, dchidT, kappa_data, λ, weights, coefs_T, coefs_Rho, ρ, T, grid_T)
    n_shells = length(coefs_T)

    @inbounds for i in range
        λ_i = λ[i]
        w_i = weights[i]
        for j in 1:n_shells
            ct = coefs_T[j]
            cr = coefs_Rho[j]
            
            it = ct.idx
            ir = cr.idx
            
            w00 = ct.w_low  * cr.w_low
            w10 = ct.w_high * cr.w_low
            w01 = ct.w_low  * cr.w_high
            w11 = ct.w_high * cr.w_high
            
            val_k = w00 * log(kappa_data[it,  ir,  i]) +
                    w10 * log(kappa_data[it+1,ir,  i]) +
                    w01 * log(kappa_data[it,  ir+1,i]) +
                    w11 * log(kappa_data[it+1,ir+1,i])

            chi[i, j] = exp(val_k) * ρ[j]
            B[i, j] = TSO.Bλ_fast(λ_i, T[j]) * w_i
            dBdT[i, j] = TSO.dBdTλ_fast(λ_i, T[j]) * w_i
            
            nT = length(grid_T)
            dChi, dT_diff = if (it > 1) && (it < nT)
                log(kappa_data[it+1, ir, i]) - log(kappa_data[it-1, ir, i]), 
                grid_T[it+1] - grid_T[it-1]
            elseif it == 1
                log(kappa_data[it+1, ir, i]) - log(kappa_data[it, ir, i]), 
                grid_T[it+1] - grid_T[it]
            else
                log(kappa_data[it, ir, i]) - log(kappa_data[it-1, ir, i]), 
                grid_T[it] - grid_T[it-1]
            end
            
            dchidT[i, j] = exp(val_k - grid_T[it]) * (dChi / dT_diff) * ρ[j]
        end
    end

    nothing
end
function _inner_compute_chunk!(range, chi, B::Nothing, dBdT::Nothing, dchidT::Nothing, kappa_data, λ, weights, coefs_T, coefs_Rho, ρ, T, grid_T)
    n_shells = length(coefs_T)

    @inbounds for i in range
        λ_i = λ[i]
        w_i = weights[i]
        for j in 1:n_shells
            ct = coefs_T[j]
            cr = coefs_Rho[j]
            
            it = ct.idx
            ir = cr.idx
            
            w00 = ct.w_low  * cr.w_low
            w10 = ct.w_high * cr.w_low
            w01 = ct.w_low  * cr.w_high
            w11 = ct.w_high * cr.w_high
            
            val_k = w00 * log(kappa_data[it,  ir,  i]) +
                    w10 * log(kappa_data[it+1,ir,  i]) +
                    w01 * log(kappa_data[it,  ir+1,i]) +
                    w11 * log(kappa_data[it+1,ir+1,i])

            chi[i, j] = exp(val_k) * ρ[j]
        end
    end

    nothing
end

# ============================================================================
# Opacity table computation with M3D
# ============================================================================

function generate_eos_identifier(feh::Float64, comp_dict::Dict{String, Float64}, alpha::Float64, vmic::Float64)
    id_parts = ["z_$(feh)_alpha_$(alpha)_vmic_$(vmic)"]
    for k in sort(collect(keys(comp_dict)))
        push!(id_parts, "$(k)_$(round(comp_dict[k], digits=3))")
    end
    return join(id_parts, "_")
end

"""
    _find_existing_eos(out_dir, feh, composition, alpha, vmic, eos_id)

Search for an existing EoS table directory. First tries matching via
`table_info.toml` files (the preferred method), then falls back to the
old name-based identifier for backwards compatibility.

Returns the path to the matching directory, or `nothing` if not found.
"""
function _find_existing_eos(out_dir::String, feh::Float64, composition::String,
                            alpha::Float64, vmic::Float64, eos_id::String,
                            version::String="")
    # 1) Preferred: search by table_info.toml metadata
    comp_str_dict = TSO.parse_composition(composition)
    match_dir = TSO.find_table_info(out_dir;
        feh=feh, alpha=alpha, composition=comp_str_dict, vmic=vmic, version=version)
    if !isnothing(match_dir)
        return match_dir
    end

    # 2) Fallback: old name-based identifier (backwards compatibility)
    full_eos_id = isempty(version) ? eos_id : "$(eos_id)_$(version)"
    eos_folder = joinpath(out_dir, full_eos_id)
    eos_files = filter(f -> startswith(f, "combined_eos_") && endswith(f, ".hdf5") && !contains(f, "eos500"), 
                        isdir(eos_folder) ? readdir(eos_folder) : String[])
    if !isempty(eos_files)
        return eos_folder
    end

    return nothing
end

function get_or_compute_eos(
    feh::Float64, 
    composition::String;
    out_dir::String,
    modelatmosfolder::String = "input/test_opac_table/",
    alpha::Float64 = 0.0,
    abund::String = "./input/abund/abund_magg",
    t_min::Float64 = 1000.0, 
    t_max::Float64 = 100000.0,
    rho_min::Float64 = 1e-18, 
    rho_max::Float64 = 1e-2,
    vmic::Float64 = 1.0,
    lambda_min::Float64 = 1000.0, 
    lambda_max::Float64 = 200000.0,
    n_lambda::Int = 100000, 
    n_t::Int = 50, 
    n_rho::Int = 50,
    nnu::Int = 32, 
    tmolim::Float64 = 100000.0,
    multi_threads::Int = 16,
    linelist_dir::String = "input/master_linelists",
    use_lambda_file::Bool = false, 
    lambda_file::String = "input/flx_wavelengths_UV.vac",
    absdat_file::String = "input/TS_absdat.dat",
    mini::Bool = false, 
    mmap::Bool = false,
    version::String = "",
    version_info::String = ""
)
    comp_dict = TSO.parse_composition(composition)
    eos_id = generate_eos_identifier(feh, comp_dict, alpha, vmic)
    existing = _find_existing_eos(out_dir, feh, composition, alpha, vmic, eos_id, version)
    if !isnothing(existing)
        info = TSO.TableInfo(existing)
        if !isnothing(info)
            print_nice("Loading EoS from $(existing) with composition $(TSO.show_composition(info))", category="Opacities", color=color_opacity, verbosity=1)
        else
            print_nice("Loading EoS from $(existing)", category="Opacities", color=color_opacity, verbosity=1)
        end
        return existing
    end

    print_nice("No EoS table found for the requested chemical composition.", category="Opacities", color=color_opacity, verbosity=2)
    cstring = TSO.show_composition(feh, alpha, comp_dict)
    print_nice("⏳ Computing EoS with composition $(cstring)", category="Opacities", color=color_opacity, verbosity=1)

    full_eos_id = isempty(version) ? eos_id : "$(eos_id)_$(version)"
    eos_folder = joinpath(out_dir, full_eos_id)

    multi_name = "model_$(full_eos_id)"
    #comp_dict_sym = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in comp_dict)
    comp_dict_sym = Dict{Symbol, Float64}(Symbol(k)=>(lowercase(k) in ["he", "li"]) ? v + feh : v for (k,v) in comp_dict)
    abund_file_path = MUST.abund_abundances(; α = alpha, comp_dict_sym..., default = abund)
    
    model = TSO.EoSTableInput(
        MUST.@in_m3dis(modelatmosfolder); 
        minT=t_min, maxT=t_max, minρ=rho_min, maxρ=rho_max, vmic=vmic, outputname=multi_name
    )

    λ_file_opt = use_lambda_file ? lambda_file : nothing
    
    print_nice("⏳ Running Multi3D...", category="Opacities", color=color_opacity, verbosity=2)
    try
        MUST.opacityTable(
            model; 
            folder=modelatmosfolder, linelist=linelist_dir, λ_file=λ_file_opt,
            λs=log(lambda_min), λe=log(lambda_max), δλ=(log(lambda_max)-log(lambda_min))/n_lambda,
            in_log=true, δlnT=(log(t_max)-log(t_min))/n_t, δlnρ=(log(rho_max)-log(rho_min))/n_rho,
            slurm=false, nν=nnu, FeH=feh, abund_file=abund_file_path, tmolim=tmolim, absdat_file=absdat_file,
            m3dis_kwargs=Dict(:threads=>multi_threads)
        )
        print_nice("✅ Multi3D completed.", category="Opacities", color=color_opacity, verbosity=2)
    catch e
        print_nice("❌ Multi3D failed. $e", category="Opacities", color=color_opacity, verbosity=1)
        error(e)
    end

    if !isdir(eos_folder)
        mkpath(eos_folder)
    end

    print_nice("⏳ Loading M3D output and collecting opacities...", category="Opacities", color=color_opacity, verbosity=2)
    run_m3d = MUST.M3DISRun("data/$(model)", read_atmos=false)
    eos, eos500, opa, scat, nan_mask_ross, nan_mask_500 = TSO.collect_opacity(run_m3d, compute_ross=true, mini=mini, mmap=mmap)
    
    print_nice("⏳ Saving variables...", category="Opacities", color=color_opacity, verbosity=2)
    eos_mono = deepcopy(eos); eos500_mono = deepcopy(eos500)
    TSO.smoothAccumulate!(eos_mono); TSO.smoothAccumulate!(eos500_mono)
    
    TSO.save(eos_mono, joinpath(eos_folder, "combined_eos_$(full_eos_id).hdf5"), nan_mask=nan_mask_ross)
    TSO.save(eos500_mono, joinpath(eos_folder, "combined_eos500_$(full_eos_id).hdf5"), nan_mask=nan_mask_500)
    TSO.save(opa, joinpath(eos_folder, "combined_opacities_$(full_eos_id).hdf5"))
    if !isnothing(scat)
        TSO.save(scat, joinpath(eos_folder, "combined_sopacities_$(full_eos_id).hdf5"))
    end

    # Save table_info.toml with the composition metadata
    comp_str_dict = Dict{String, Float64}(string(k) => v for (k, v) in comp_dict)
    info = TSO.TableInfo(
        feh=feh, alpha=alpha, composition=comp_str_dict,
        vmic=vmic, version=version,
        abund=abund,
        t_min=t_min, t_max=t_max, rho_min=rho_min, rho_max=rho_max,
        lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=n_lambda,
        n_t=n_t, n_rho=n_rho, nnu=nnu, tmolim=tmolim,
        linelist_dir=linelist_dir, use_lambda_file=use_lambda_file,
        lambda_file=lambda_file, absdat_file=absdat_file,
        mini=mini, mmap=mmap, version_info=version_info
    )
    TSO.save(info, eos_folder)

    # delete the multi3d files
    rm(MUST.@in_tumult("data/$(model)"), recursive=true)
    
    return eos_folder
end