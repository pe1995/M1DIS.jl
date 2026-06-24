const k_B = 1.380649e-16  
const h   = 6.626070e-27  
const m_u = 1.660539e-24
const c_light = 2.99792458e10
const σ_SB = 5.670374e-5
const verbose = Ref{Int}(1)

const color_star = :light_red
const color_planet = :light_green
const color_opacity = :light_cyan
const color_spectrum = :light_magenta
const color_messages = Ref{Symbol}(color_star)

function print_nice(s; category="", verbosity=verbose[], kwargs...)
    if verbose[] >= verbosity
        if category == ""
            printstyled(s, "\n"; kwargs...)
        else 
            printstyled("[ $(category): "; bold=true, kwargs...)
            printstyled(s, "\n")
        end
    end
end

macro verbose_info(level, args...)
    return quote
        if verbose[] >= $(esc(level))
            @info($(esc.(args)...))
        end
    end
end

macro verbose_warn(level, args...)
    return quote
        if verbose[] >= $(esc(level))
            @warn($(esc.(args)...))
        end
    end
end

# helper function for nothing
import Base.similar
Base.similar(::Nothing) = nothing

