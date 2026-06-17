# ==============================================================================
# Data structures
# ==============================================================================

mutable struct Atmosphere{T <: AbstractFloat}
    # Base Parameters
    T_eff::T              # Effective Temperature [K] 
    z::Vector{T}          # Height (Size: D)
    tau::Vector{T}        # Reference optical depth (Size: D)
    tau_lambda::Matrix{T} # Reference optical depth (Size: Nf, D)
    rho::Vector{T}        # Density [g/cm^3] (Size: D)
    Temp::Vector{T}       # Temperature (Size: D)
    P_gas::Vector{T}      # Gas Pressure [dyne/cm^2] (Size: D)
    mu::Vector{T}         # Angle cosines (Size: Na)
    w_mu::Vector{T}       # Angle weights (Normalized to sum=1)
    
    # Opacities & Source
    chi::Matrix{T}        # Opacity (Nf x D)
    chi_scat::Union{Matrix{T}, Nothing}   # Scattering opacity (Nf x D)
    chi_ref::Vector{T}    # Reference opacity (Size: D)
    B::Matrix{T}          # Planck function (Nf x D)
    dBdT::Matrix{T}       # Derivative of B (Nf x D)
    dchidT::Matrix{T}     # Derivative of opacity (Nf x D)
    dchidT_scat::Union{Matrix{T}, Nothing} # Derivative of opacity (Nf x D)
    I_top::Vector{T}      # External Irradiation (Size: Nf)
    
    # Convection
    F_conv::Vector{T}              # Convective Flux (Size: D) 
    dFconv_dT::Vector{T}           # Jacobian term: dF_conv / dT_local (Size: D) 
    v_conv::Vector{T}              # Convective velocity (Size: D)
    P_turb::Vector{T}              # Turbulent pressure [dyne/cm^2] (Size: D)
    
    # Radiation Transport Outputs
    J_bol::Vector{T}      # Bolometric Mean Intensity (Size: D)
    F_bol::Vector{T}      # Bolometric Flux on the nodes (Size: D)
    F_rad::Vector{T}      # Radiative Flux at the cell interfaces (Size: D)
	g_rad::Vector{T}      # Radiative Acceleration (Size: D) [cm/s^2] 
    P_rad::Vector{T}      # Radiative Pressure (Size: D) [dyne/cm^2]
    Q_rad::Vector{T}      # Radiative Heating [erg/cm^3/s] (Size: D)
    J_raw                 # Specific Intensity J(mu) (Nf x Na x D)
    
    # Convergence Tracking
    dT::Vector{T}         # Temperature Correction (Size: D)
    F_total::Vector{T}    # Total Flux = F_rad + F_conv (Size: D)
    F_err_rel::Vector{T}  # Relative Flux Error (Size: D)
    irrad_iso::Bool       # Isotropic irradiation
    irrad_mu::T           # Direction cosine of incident radiation
end

"""
    Atmosphere(tau, Nf; T_eff, n_angles=4, mu=nothing, w_mu=nothing,
               with_scattering=false, irrad_iso=false, irrad_mu=0.5)

Allocate a 1D model atmosphere. All physics arrays are **zero-initialized**;
populate them after construction and call `update!(atm)` to derive `tau_lambda`.

# Positional arguments
- `tau`  — reference optical depth grid; sets the depth dimension `D = length(tau)`
- `Nf`   — number of frequency / opacity bins

# Keyword arguments
- `T_eff`           — effective temperature [K]
- `n_angles`        — number of Gauss-Legendre angle points (default: 4);
                      ignored when `mu` is supplied explicitly
- `mu`              — custom angle cosines; overrides `n_angles`
- `w_mu`            — custom quadrature weights (required when `mu` is given)
- `with_scattering` — allocate `chi_scat` / `dchidT_scat` buffers (default: `false`)
- `irrad_iso`       — isotropic irradiation flag (default: `false`)
- `irrad_mu`        — direction cosine of incident irradiation (default: `0.5`)

# Typical usage
```julia
atm = Atmosphere(τ_grid, Nf; T_eff=5777.0)

atm.Temp .= T_initial
atm.rho  .= ρ_initial
atm.z    .= z_initial
atm.P_gas .= P_initial
compute_opacities!(atm.chi, atm.chi_ref, atm.B, atm.dBdT, atm.dchidT, eos, opa, atm.Temp, atm.rho)
update!(atm)   # computes tau_lambda from z and chi
```
"""
function Atmosphere(tau::AbstractVector, Nf::Int;
                    T_eff::Real,
                    n_angles::Int                       = 4,
                    mu::Union{AbstractVector, Nothing}  = nothing,
                    w_mu::Union{AbstractVector, Nothing} = nothing,
                    with_scattering::Bool               = false,
                    irrad_iso::Bool                     = false,
                    irrad_mu::Real                      = 0.5)

    FT = eltype(tau)
    D  = length(tau)

    # Angular quadrature
    # Default: Gauss-Legendre mapped to (0, 1]; custom mu/w_mu override this.
    _mu, _w_mu = if isnothing(mu)
        x, w = gausslegendre(n_angles)
        FT.(x ./ 2 .+ 0.5), FT.(w ./ 2)
    else
        @assert !isnothing(w_mu) "w_mu must be provided when mu is given"
        FT.(mu), FT.(w_mu)
    end
    # Normalize weights to sum = 1
    s = sum(_w_mu)
    s > 0 && (_w_mu ./= s)

    # Optional scattering buffers
    chi_scat    = with_scattering ? zeros(FT, Nf, D) : nothing
    dchidT_scat = with_scattering ? zeros(FT, Nf, D) : nothing

    return Atmosphere{FT}(
        FT(T_eff),
        # geometry
        zeros(FT, D),          # z
        copy(FT.(tau)),        # tau   (deep-copied; caller's array is untouched)
        zeros(FT, Nf, D),      # tau_lambda  (filled by update!)
        # thermodynamics
        zeros(FT, D),          # rho
        zeros(FT, D),          # Temp
        zeros(FT, D),          # P_gas
        # angles (deep-copied)
        copy(_mu), copy(_w_mu),
        # opacities & source (filled by compute_opacities!)
        zeros(FT, Nf, D),      # chi
        chi_scat,              # chi_scat
        zeros(FT, D),          # chi_ref
        zeros(FT, Nf, D),      # B
        zeros(FT, Nf, D),      # dBdT
        zeros(FT, Nf, D),      # dchidT
        dchidT_scat,           # dchidT_scat
        zeros(FT, Nf),         # I_top
        # convection
        zeros(FT, D),          # F_conv
        zeros(FT, D),          # dFconv_dT
        zeros(FT, D),          # v_conv
        zeros(FT, D),          # P_turb
        # RT outputs
        zeros(FT, D),          # J_bol
        zeros(FT, D),          # F_bol
        zeros(FT, D),          # F_rad
        zeros(FT, D),          # g_rad
        zeros(FT, D),          # P_rad
        zeros(FT, D),          # Q_rad
        nothing,               # J_raw
        # convergence
        zeros(FT, D),          # dT
        zeros(FT, D),          # F_total
        zeros(FT, D),          # F_err_rel
        irrad_iso,
        FT(irrad_mu)
    )
end

"""
    populate!(atm::Atmosphere; kwargs...) → atm

Set one or more fields of `atm` in-place using keyword arguments.

- For **array fields** the existing buffer is filled with `.=` (no allocation).
- For **scalar / Nothing fields** the value is set with `setfield!`.

After calling `populate!` with geometry or opacity data, call `update!(atm)`
to recompute `tau_lambda`.

# Example
```julia
populate!(atm;
    Temp=T_initial, rho=ρ_initial, z=z_initial, P_gas=P_initial,
    chi=chi_arr, chi_ref=chi_ref_arr,
    B=S_arr, dBdT=dSdT_arr, dchidT=dchidT_arr,
)
update!(atm)
```
"""
function populate!(atm::Atmosphere; kwargs...)
    for (field, val) in kwargs
        current = getfield(atm, field)
        if current isa AbstractArray
            current .= val          # fill pre-allocated buffer, no allocation
        else
            setfield!(atm, field, val)   # scalar, Nothing, or type change
        end
    end
    return atm
end

function update!(atm::Atmosphere{T}; sync_opacities::Bool=true, sync_geometry::Bool=true, sync_angles::Bool=false) where T
    D = length(atm.tau)
	Nf = size(atm.chi, 1)

    # Recompute frequency-dependent optical depth
    if sync_opacities || sync_geometry
        Threads.@threads for f in 1:Nf
             compute_τ!(view(atm.tau_lambda, f, :); z=atm.z, ρκ=view(atm.chi, f, :))
        end
    end
    
    # Re-normalize angular weights (usually only needed if angles change dynamically)
    if sync_angles
        total = sum(atm.w_mu)
        if total > 0
            atm.w_mu ./= total
        end
    end

    return nothing
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