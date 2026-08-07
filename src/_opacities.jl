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
