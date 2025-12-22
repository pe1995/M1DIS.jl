"""
    update_z_grid!(z; T, ρ, τ, eos)

Recompute z scale for a given T, ρ structure on fixed τ grid.
"""
function update_z_grid!(z; T, ρ, τ, eos)
    z[1] = 0.0 
    @inbounds for i in 1:(length(z)-1)
        T_mid = 0.5 * (T[i] + T[i+1])
        ρ_mid = 0.5 * (ρ[i] + ρ[i+1])
        κ_R = exp(lookup(eos, :lnRoss, log(ρ_mid), log(T_mid)))
        dτ = τ[i+1] - τ[i]
        dz = dτ / (κ_R * ρ_mid)
        z[i+1] = z[i] + dz
    end
    z
end

"""
    update_τ_grid!(τ; T, ρ, z, eos)

Recompute τ scale for a given T, ρ structure on fixed z grid.
"""
function update_τ_grid!(τ; T, ρ, z, eos)
    ρκ = exp.(lookup(eos, :lnRoss, log.(ρ), log.(T)))
    ρκ .= exp.(log.(ρ)) .* ρκ

    compute_τ_grid!(τ; z=z, ρκ=ρκ)
end

function compute_τ_grid!(τ; z, ρκ)
    # Integrate: τ(z) = [ ∫ ρκ dz ]_z0 ^z
    @inbounds for j in eachindex(τ)
        if j==1 
            τ[1] = 0 + (z[2] - z[1]) * 0.5 * (ρκ[j])
        else
            τ[j] = τ[j-1] + (z[j] - z[j-1]) * 0.5 * (ρκ[j] + ρκ[j-1])
        end
    end
end










#= HE solver =#

"""
    hydrostatic_equilibrium!(T_ip, g_rad_ip, g_turb_ip; g, eos)

Generate the hydrostatic equilibrium function. Interpolated T, g_rad and g_turb
functions are needed to interpolate the structure to any given τ. Note that the
solver works with lnP instead of P itself.
"""
function hydrostatic_equilibrium!(T_ip, g_rad_ip, g_turb_ip; g, eos)
    function HE!(du, u, p, τ)
        lnP = u[1]
		lgt = log10(τ)
		
        lnT = T_ip(lgt)
        g_rad  = g_rad_ip(lgt)
        g_turb = g_turb_ip(lgt)

        g_eff = max(g - g_rad - g_turb, 0.0)

        lnρ = TSO.extended_lookup(eos, :lnRho, lnP, lnT)
        κ_ross = exp(TSO.extended_lookup(eos, :lnRoss, lnρ, lnT))

        P = exp(lnP)
        du[1] = g_eff / (κ_ross * P)
    end
    return HE!
end

function update_hydrostatic!(P, ρ, z, T, g_rad, g_turb, τ_grid; eos, logg)
    # prepare interpolations in log10(τ)
    T_ip = linear_interpolation(log10.(τ_grid), log.(T))
    g_rad_ip = linear_interpolation(log10.(τ_grid), g_rad)
    g_turb_ip = linear_interpolation(log10.(τ_grid), g_turb)

    g_const = exp10(logg)

    # top boundary
    τ_top = τ_grid[1]
    T_top = T[1]
    g_eff_top = max(g_const - g_rad[1] - g_turb[1], 0.0)
    lnP_top = lnP_boundary(T_top, g_eff_top, eos, τ_top)

    # solve HE from top to bottom on the rosseland optical depth scale
    structure_eq = hydrostatic_equilibrium!(
        T_ip, g_rad_ip, g_turb_ip; g=g_const, eos=eos
    )
    u0 = [lnP_top]
    tspan = (τ_grid[1], τ_grid[end])
    prob = ODEProblem(structure_eq, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=τ_grid)

    # extract the pressure and density; compute new z scale
	P .= [u[1] for u in sol.u] .|> exp
	ρ .= [TSO.extended_lookup(eos,:lnRho,log(pi),log(ti)) for (pi,ti) in zip(P,T)] .|> exp
	update_z_grid!(z, T=T, ρ=ρ, τ=τ_grid, eos=eos.eos);
end
