# ============================================================================
# HE solver
# ============================================================================

"""
    hydrostatic_equilibrium!(T_ip, g_rad_ip, g_turb_ip; g, eos)

Generate the hydrostatic equilibrium function. Interpolated T, g_rad and g_turb
functions are needed to interpolate the structure to any given τ. Note that the
solver works with lnP instead of P itself.
"""
function hydrostatic_equilibrium!(T_ip, P_rad_ip, P_turb_ip; g, eos)
    ln10 = log(10.0)
    g_ln10 = g * ln10

    function HE!(du, u, p, lgt)
        lnP = u[1]
		#lgt = log10(τ)
		
        lnT = T_ip(lgt)
        P_rad  = exp(P_rad_ip(lgt))
        P_turb = exp(P_turb_ip(lgt))
        P_tot = exp(lnP)

        P_gas = max(P_tot - P_rad - P_turb, 1e-30)
        _, lnκ_ross = sample(eos, (:lnRho, :lnRoss), log(P_gas), lnT)

        ln_tau = lgt * ln10
        du[1] = g_ln10 * exp(ln_tau - lnκ_ross - lnP)

        #du[1] = g/ (exp(lnκ_ross) * P_tot)
    end
    return HE!
end

function update_hydrostatic!(P_gas, ρ, z, T, P_turb, P_rad, τ_grid; eos, logg)
    # Prepare interpolations in log10(τ)
    # input P array is P_gas, however the solver expects P_tot
    lgt = log10.(τ_grid)
    T_ip = linear_interpolation(lgt, log.(T))
    P_turb_ip = linear_interpolation(lgt, log.(max.(P_turb, 1e-30)))
    P_rad_ip = linear_interpolation(lgt, log.(max.(P_rad, 1e-30)))

    g_const = exp10(logg)

    τ_top = τ_grid[1]
    T_top = T[1]
    lnP_top = lnP_boundary(T_top, g_const, eos, τ_top, P_guess=max(1.0, P_gas[1]))
    u0 = [log(exp(lnP_top) + P_rad[1] + P_turb[1])]
    #tspan = (τ_grid[1], τ_grid[end])
    tspan = (lgt[1], lgt[end])

    structure_eq = hydrostatic_equilibrium!(
        T_ip, P_rad_ip, P_turb_ip; g=g_const, eos=eos
    )
    prob = ODEProblem(structure_eq, u0, tspan)
    #sol = solve(prob, Tsit5(), saveat=τ_grid)
    sol = solve(prob, Tsit5(), saveat=lgt, reltol=1e-8, abstol=1e-8, dtmax=0.05)

    for i in eachindex(T)
        P_tot = exp(sol.u[i][1]) # The solver returns ln(P_tot)
        P_gas[i] = max(P_tot - P_rad[i] - P_turb[i], 1e-30)
        ρ[i], = sample(eos, (:lnRho,), log(P_gas[i]), log(T[i])) .|> exp
    end
    
    update_z_grid!(z, T=T, ρ=ρ, τ=τ_grid, eos=eos.eos)
end

# ============================================================================
# Grid updates
# ============================================================================

"""
    update_z_grid!(z; T, ρ, τ, eos)

Recompute z scale for a given T, ρ structure on fixed τ grid.
"""
function update_z_grid!(z; T, ρ, τ, eos)
    z[1] = 0.0 
    @inbounds for i in 1:(length(z)-1)
        T_mid = 0.5 * (T[i] + T[i+1])
        ρ_mid = 0.5 * (ρ[i] + ρ[i+1])
        κ_R = exp(sample(eos, (:lnRoss,), log(ρ_mid), log(T_mid))[1])
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
