const k_B = 1.380649e-16  
const h   = 6.626070e-27  
const m_u = 1.660539e-24
const c_light = 2.99792458e10
const σ_SB = 5.670374e-5
const verbose = Ref{Int}(1)

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

