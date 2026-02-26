# ============================================================================
# Global timers
# ============================================================================

const generalTimer = TimerOutput()
const initialization_time = Ref(false)
const mixing_length_time = Ref(false)
const radiation_transfer_time = Ref(false)
const hydrostatic_time = Ref(false)
const compute_opacities_time = Ref(false)
const update_atmosphere_time = Ref(false)
const solve_RT_time = Ref(false)
const relaxation_time = Ref(false)

const timers = [
    initialization_time,
    mixing_length_time,
    radiation_transfer_time,
    compute_opacities_time,
    update_atmosphere_time,
    solve_RT_time,
    hydrostatic_time,
    relaxation_time
]

# ============================================================================
# Activate/Deactivate timers
# ============================================================================

activate_timing!() = activate_timing!.(timers)
activate_timing!(t) = begin
    reset_timer!(generalTimer)
    t[] = true
end

deactivate_timing!() = deactivate_timing!.(timers)
deactivate_timing!(t) = begin
    reset_timer!(generalTimer)
    t[] = false
end

start_timing!(t=generalTimer) = reset_timer!(t) 
end_timing!(t=generalTimer) = begin
    println("")
    show(t)
    println("") 

    t
end

timer() = generalTimer

# ============================================================================
# Optional timing Macro to add to function calls
# ============================================================================

macro optionalTiming(name, exp)
    name_e = esc(name)
    ex = esc(exp)
    name_string = "$(name)"
    quote
        if $(name_e)[]
            @timeit generalTimer $(name_string) begin
                $(ex)
            end
        else
            $(ex)
        end
    end
end