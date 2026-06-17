# ============================================================================
# Atmosphere from MUST.Box
# ============================================================================

function Atmosphere(b::MUST.Box, eos, opacity; scattering=nothing, T_eff=b.parameter.teff, downsample=1)
    b = deepcopy(b)
    MUST.flip!(b, depth=true)
    T   = MUST.profile(TSO.mean, b, :z, :T)[2][1:downsample:end]
    ρ   = MUST.profile(TSO.mean, b, :z, :log10d)[2][1:downsample:end] .|> exp10
    z   = MUST.profile(TSO.mean, b, :z, :z)[2][1:downsample:end]
    τ   = MUST.profile(TSO.mean, b, :z, :log10τ_ross)[2][1:downsample:end] .|> exp10
    Pg  = MUST.profile(TSO.mean, b, :z, :log10Pg)[2][1:downsample:end] .|> exp10

    chi, chi_ref, S, dSdT, dchidT =
        opacity.binned ? compute_opacities(eos, opacity, T, ρ) :
                         compute_opacities_chunked(eos, opacity, T, ρ)
    chi_scat = isnothing(scattering) ? nothing :
        compute_opacities_chunked(eos, scattering, T, ρ, opacity_only=true)[1]

    Nf  = size(chi, 1)
    atm = Atmosphere(τ, Nf; T_eff=T_eff, with_scattering=!isnothing(chi_scat))

    populate!(atm;
        Temp=T, rho=ρ, z=z, P_gas=Pg,
        chi=chi, chi_ref=chi_ref,
        B=S, dBdT=dSdT, dchidT=dchidT,
    )
    isnothing(chi_scat) || populate!(atm; chi_scat=chi_scat)
    update!(atm)

    return atm
end