# ============================================================================
# Data structures
# ============================================================================

mutable struct Atmosphere{T <: AbstractFloat}
    T_eff::T
    z::Vector{T}
    tau::Vector{T}
    tau_lambda::Matrix{T}
    rho::Vector{T}
    Temp::Vector{T}
    P_gas::Vector{T}
    mu::Vector{T}
    w_mu::Vector{T}

    chi::Matrix{T}
    chi_scat::Union{Matrix{T}, Nothing}
    chi_ref::Vector{T}
    B::Matrix{T}
    dBdT::Matrix{T}
    dchidT::Matrix{T}
    dchidT_scat::Union{Matrix{T}, Nothing}
    I_top::Vector{T}

    F_conv::Vector{T}
    dFconv_dT::Vector{T}
    v_conv::Vector{T}
    P_turb::Vector{T}

    J_bol::Vector{T}
    F_bol::Vector{T}
    F_rad::Vector{T}
	g_rad::Vector{T}
    P_rad::Vector{T}
    Q_rad::Vector{T}
    F_beam::Vector{T}
    J_raw

    dT::Vector{T}
    F_total::Vector{T}
    F_err_rel::Vector{T}
    irrad_iso::Bool
    irrad_mu::T
    f_redist::T
end

function Atmosphere(tau::AbstractVector, Nf::Int;
                    T_eff::Real,
                    n_angles::Int                       = 4,
                    mu::Union{AbstractVector, Nothing}  = nothing,
                    w_mu::Union{AbstractVector, Nothing} = nothing,
                    with_scattering::Bool               = false,
                    irrad_iso::Bool                     = false,
                    irrad_mu::Real                      = 0.5,
                    f_redist::Real                      = 0.25)

    FT = eltype(tau)
    D  = length(tau)

    _mu, _w_mu = if isnothing(mu)
        x, w = gausslegendre(n_angles)
        FT.(x ./ 2 .+ 0.5), FT.(w ./ 2)
    else
        @assert !isnothing(w_mu) "w_mu must be provided when mu is given"
        FT.(mu), FT.(w_mu)
    end
    s = sum(_w_mu)
    s > 0 && (_w_mu ./= s)

    chi_scat    = with_scattering ? zeros(FT, Nf, D) : nothing
    dchidT_scat = with_scattering ? zeros(FT, Nf, D) : nothing

    return Atmosphere{FT}(
        FT(T_eff),
        zeros(FT, D),
        copy(FT.(tau)),
        zeros(FT, Nf, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        copy(_mu), copy(_w_mu),
        zeros(FT, Nf, D),
        chi_scat,
        zeros(FT, D),
        zeros(FT, Nf, D),
        zeros(FT, Nf, D),
        zeros(FT, Nf, D),
        dchidT_scat,
        zeros(FT, Nf),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        nothing,
        zeros(FT, D),
        zeros(FT, D),
        zeros(FT, D),
        irrad_iso,
        FT(irrad_mu),
        FT(f_redist)
    )
end

function populate!(atm::Atmosphere; kwargs...)
    for (field, val) in kwargs
        current = getfield(atm, field)
        if current isa AbstractArray
            current .= val
        else
            setfield!(atm, field, val)
        end
    end
    return atm
end

function update!(atm::Atmosphere{T}; sync_opacities::Bool=true, sync_geometry::Bool=true, sync_angles::Bool=false) where T
    D = length(atm.tau)
	Nf = size(atm.chi, 1)

    if sync_opacities || sync_geometry
        Threads.@threads for f in 1:Nf
             compute_τ!(view(atm.tau_lambda, f, :); z=atm.z, ρκ=view(atm.chi, f, :))
        end
        beam_flux!(atm)
    end

    if sync_angles
        total = sum(atm.w_mu)
        if total > 0
            atm.w_mu ./= total
        end
    end

    return nothing
end

function beam_flux!(atm::Atmosphere)
    fill!(atm.F_beam, zero(eltype(atm.F_beam)))
    mu_star = atm.irrad_mu
    @inbounds for f in axes(atm.chi, 1)
        atm.I_top[f] > 0.0 || continue
        F0 = atm.f_redist * π * atm.I_top[f]
        for d in eachindex(atm.F_beam)
            atm.F_beam[d] += F0 * exp(-atm.tau_lambda[f, d] / mu_star)
        end
    end
    atm.F_beam
end

# ============================================================================
# Compute formation height
# ============================================================================

function formation_height(atm::M1DIS.Atmosphere; closest=false)
    lgt = log10.(atm.tau)
    fh = similar(atm.tau_lambda, size(atm.tau_lambda, 1))

    if closest
        for l in axes(atm.tau_lambda, 1)
            idx = argmin(abs.(atm.tau_lambda[l, :] .- 1.0))
            fh[l] = -lgt[idx]
        end
    else
        for l in axes(atm.tau_lambda, 1)
            fh[l] = -MUST.linear_interpolation(log10.(atm.tau_lambda[l, :]), lgt, extrapolation_bc=MUST.Flat())(0.0)
        end
    end

    fh
end

function formation_source_function(atm::M1DIS.Atmosphere; closest=false)
    lgt = log10.(atm.tau)
    sf = similar(atm.tau_lambda, size(atm.tau_lambda, 1))

    if closest
        for l in axes(atm.tau_lambda, 1)
            idx = argmin(abs.(atm.tau_lambda[l, :] .- 1.0))
            sf[l] = log10.(atm.B[l, idx])
        end
    else
        for l in axes(atm.tau_lambda, 1)
            log_tau_l = log10.(atm.tau_lambda[l, :])
            sf[l] = MUST.linear_interpolation(log_tau_l, log10.(atm.B[l, :]), extrapolation_bc=MUST.Flat())(0.0)
        end
    end

    return sf
end

function formation_opacity(atm::M1DIS.Atmosphere; closest=false)
    lgt = log10.(atm.tau)
    sf = similar(atm.tau_lambda, size(atm.tau_lambda, 1))

    if closest
        for l in axes(atm.tau_lambda, 1)
            idx = argmin(abs.(atm.tau_lambda[l, :] .- 1.0))
            sf[l] = log10.(atm.chi[l, idx])
        end
    else
        for l in axes(atm.tau_lambda, 1)
            log_tau_l = log10.(atm.tau_lambda[l, :])
            sf[l] = MUST.linear_interpolation(log_tau_l, log10.(atm.chi[l, :]), extrapolation_bc=MUST.Flat())(0.0)
        end
    end

    return sf
end

function opacity_at(atm::M1DIS.Atmosphere, logtau; closest=false)
    lgt = log10.(atm.tau)
    op = similar(atm.chi, size(atm.chi, 1))
    lgo = similar(atm.chi, size(atm.chi, 2))

    if closest
        idx = argmin(abs.(lgt .- logtau))
        for l in axes(atm.chi, 1)
            op[l] = log10.(atm.chi[l, idx])
        end
    else
        for l in axes(atm.chi, 1)
            @. lgo = log10(atm.chi[l, :])
            op[l] = MUST.linear_interpolation(lgt, lgo, extrapolation_bc=MUST.Flat())(logtau)
        end
    end

    op
end
