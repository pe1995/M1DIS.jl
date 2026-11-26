using PrecompileTools

@setup_workload begin
    println("Setting up M1DIS precompilation workload...")

    eos_dir = joinpath(@__DIR__, "..", "data", "magg_mini_v1.0")

    if isdir(eos_dir)
        table_mini = TSO.EquationOfState(eos_dir; mini=true)
        table_extended = TSO.EquationOfState(eos_dir)

        M1DIS.USE_RT_THREADS[] = false
        M1DIS.verbose[] = 0
        @compile_workload begin
            println("Compiling M1DIS workload...")
            try
                atmosphere(
                    T_eff = 5777.0,
                    logg = 4.44,
                    α_MLT = 1.5,
                    maxiter = 1,
                    table = table_mini,
                    damping = 0.1,
                    τ = 10.0 .^ range(-4.0, 1.0, length=30),
                    use_threads = true,
                    feautrier = true,
                )
                println("✅ Precompiled without source function.")
            catch e
                @warn "❌ Precompilation M1DIS workload failed without source function." exception=(e, catch_backtrace())
            end

            try
                atmosphere(
                    T_eff = 5777.0,
                    logg = 4.44,
                    α_MLT = 1.5,
                    maxiter = 1,
                    table = table_extended,
                    damping = 0.1,
                    τ = 10.0 .^ range(-4.0, 1.0, length=30),
                    use_threads = true,
                    feautrier = true,
                )
                println("✅ Precompiled with source function.")
            catch e
                @warn "❌ Precompilation M1DIS workload failed with source function." exception=(e, catch_backtrace())
            end
        end
        M1DIS.verbose[] = 1
        M1DIS.USE_RT_THREADS[] = true
    else
        @warn "Precompilation data not found at $eos_dir. Skipping M1DIS precompilation."
    end
end
