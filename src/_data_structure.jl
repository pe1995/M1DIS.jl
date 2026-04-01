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
    eta::Matrix{T}        # Opacity ratio chi / chi_ref
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

function Atmosphere(; T_eff::T, z::Vector{T}, tau::Vector{T}, rho::Vector{T}, Temp::Vector{T}, P_gas::Vector{T},
                    mu::Vector{T}, w_mu::Vector{T}, 
                    chi::Matrix{T}, chi_ref::Vector{T}, 
					B::Matrix{T}, dBdT::Matrix{T}, dchidT::Matrix{T},
					I_top::Union{Vector{T}, Nothing}=nothing, chi_scat::Union{Matrix{T}, Nothing}=nothing, irrad_iso::Bool=false, irrad_mu::T=1.0/sqrt(3.0)) where T
    D = length(tau)
    Nf = size(chi, 1) 
    Na = length(mu)
    
    # Normalize Angle Weights
    total_w_mu = sum(w_mu)
    w_mu_norm = (total_w_mu > 0) ? w_mu ./ total_w_mu : deepcopy(w_mu)
    
    # Eta
    eta = zeros(T, Nf, D)
    for d in 1:D
		ref = max(chi_ref[d], 1e-30)
		for f in 1:Nf
            eta[f, d] = chi[f, d] / ref
        end
    end

    # Compute tau_lambda
    tau_lambda = zeros(T, Nf, D)
    Threads.@threads for f in 1:Nf
        compute_τ!(view(tau_lambda, f, :); z=z, ρκ=chi[f,:])
    end
    
    I_top_val = isnothing(I_top) ? zeros(T, Nf) : deepcopy(I_top)
    chi_scat_val = isnothing(chi_scat) ? nothing : deepcopy(chi_scat)
    
    # Allocation of Internal Storage
    # Memory for convection
    F_conv         = zeros(T, D)
    dFconv_dT      = zeros(T, D)
    v_conv         = zeros(T, D)
    P_turb         = zeros(T, D)
    
    # Memory for RT outputs
    J_bol      = zeros(T, D)
	F_bol      = zeros(T, D)
	F_rad      = zeros(T, D)
    g_rad      = zeros(T, D)
    P_rad      = zeros(T, D)
    Q_rad      = zeros(T, D)
    dT         = zeros(T, D)
    J_raw_init = nothing 

    # Memory for convergence evaluation
    F_total   = zeros(T, D)
    F_err_rel = zeros(T, D)
    
    return Atmosphere{T}(
        T_eff, deepcopy(z), deepcopy(tau), tau_lambda, deepcopy(rho), deepcopy(Temp), deepcopy(P_gas),
        deepcopy(mu), w_mu_norm, 
        deepcopy(chi), chi_scat_val, deepcopy(chi_ref), deepcopy(B), deepcopy(dBdT), deepcopy(dchidT), eta, I_top_val, 
        F_conv, dFconv_dT, v_conv, P_turb,
        J_bol, F_bol, F_rad, g_rad, P_rad, Q_rad, J_raw_init,
        dT, F_total, F_err_rel, irrad_iso, irrad_mu
    )
end

function update!(atm::Atmosphere{T}; sync_opacities::Bool=true, sync_geometry::Bool=true, sync_angles::Bool=false) where T
    D = length(atm.tau)
	Nf = size(atm.chi, 1)

	# Recompute Eta 
    if sync_opacities
        @inbounds for d in 1:D
			ref = max(atm.chi_ref[d], 1e-30)
            for f in 1:Nf
                atm.eta[f, d] = atm.chi[f, d] / ref
            end
        end
    end

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

function formation_height(atm::M1DIS.Atmosphere; closest=true)
    lgt = log10.(atm.tau)
    fh = similar(atm.tau_lambda, size(atm.tau_lambda, 1))

    if closest
        for l in axes(atm.tau_lambda, 1)
            idx = argmin(abs.(atm.tau_lambda[l, :] .- 1.0))
            fh[l] = -lgt[idx]
        end
    else
        for l in axes(atm.tau_lambda, 1)
            fh[l] = -MUST.linear_interpolation(log10.(atm.tau_lambda[l, :]), lgt, extrapolation_bc=MUST.Line())(0.0)
        end
    end
   
    fh
end