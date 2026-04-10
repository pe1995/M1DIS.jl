# ============================================================================
# Atmosphere from MUST.Box
# ============================================================================

function Atmosphere(b::MUST.Box, eos, opacity; scattering=nothing, T_eff=b.parameter.teff, downsample=1)
    b = deepcopy(b)
    MUST.flip!(b, depth=true)
    T = reshape(b.data[:T], :)[1:downsample:end]
    ρ = reshape(b.data[:d], :)[1:downsample:end]
    z = reshape(b.z, :)[1:downsample:end]
    τ = reshape(b.data[:τ_ross], :)[1:downsample:end]
    Pg = reshape(b.data[:Pg], :)[1:downsample:end]
    
    μ_angles, μ_weights = generate_mu_grid(4)
    chi, chi_ref, S, dSdT, dchidT = opacity.binned ? compute_opacities(eos, opacity, T, ρ) : compute_opacities_chunked(eos, opacity, T, ρ)
	chi_scat  = !isnothing(scattering) ? compute_opacities_chunked(eos, scattering, T, ρ, opacity_only=true)[1] : nothing

    a = Atmosphere(
        T_eff=T_eff, 
        z=z, 
        tau=τ, 
        rho=ρ, 
        Temp=T, 
        P_gas=Pg, 
        mu=μ_angles, 
        w_mu=μ_weights, 
        chi=chi, 
        chi_scat=chi_scat,
        chi_ref=chi_ref, 
        B=S, 
        dBdT=dSdT,
        dchidT=dchidT
    )
    #update!(a, sync_opacities=true, sync_geometry=true, sync_angles=true)
    return a
end