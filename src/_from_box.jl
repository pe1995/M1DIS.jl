# ============================================================================
# Atmosphere from MUST.Box
# ============================================================================

function Atmosphere(b::MUST.Box, eos, opacity; scattering=nothing, T_eff=b.parameter.teff, downsample=1)
    b = deepcopy(b)
    MUST.flip!(b, depth=true)
    T = MUST.profile(TSO.mean, b, :z, :T)[2][1:downsample:end]
    ρ = MUST.profile(TSO.mean, b, :z, :log10d)[2][1:downsample:end] .|> exp10
    z = MUST.profile(TSO.mean, b, :z, :z)[2][1:downsample:end]
    τ = MUST.profile(TSO.mean, b, :z, :log10τ_ross)[2][1:downsample:end] .|> exp10
    Pg = MUST.profile(TSO.mean, b, :z, :log10Pg)[2][1:downsample:end] .|> exp10
    
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
    return a
end