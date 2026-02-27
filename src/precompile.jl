using PrecompileTools

@setup_workload begin
    println("Setting up M1DIS precompilation workload...")

    # We use the mini opacity table that is only 13MB in size to avoid
    # downloading or checking out large LFS files during precompilation.
    eos_dir = joinpath(@__DIR__, "..", "data", "magg_mini_v1.0")
    
    # We only precompile if the dummy table exists locally to avoid breaking 
    # CI or user installs that don't have this folder yet.
    if isdir(eos_dir)
        eos_file = MUST.glob("*_eos_*.hdf5", eos_dir)[1]
        eos500_file = MUST.glob("*_eos500_*.hdf5", eos_dir)[1]
        opa_file = MUST.glob("*_opacities_*.hdf5", eos_dir)[1]

        eos = TSO.reload(eos_file)
        eos500 = TSO.reload(eos500_file)
        opa_mini = TSO.reload(TSO.MiniOpacityTable, opa_file)
        opa_extended = TSO.extended(TSO.reload(opa_file, mmap=true), binned=false)

        M1DIS.verbose[] = 0
        @compile_workload begin
            println("Compiling M1DIS workload...")
            try
                atmosphere(
                    T_eff = 5777.0,
                    logg = 4.44,
                    v_mic = 1.0,
                    α_MLT = 1.5,
                    maxiter = 1,
                    eos = eos,
                    opacity = opa_mini,
                    damping = 0.1,
                    τ = 10.0 .^ range(-4.0, 1.0, length=30),
                    use_threads = true,
                    feutrier = true,
                )
                println("✅ Precompiled with MiniOpacityTable.")
            catch e
                @warn "❌ Precompilation M1DIS workload failed with the MiniOpacityTable." exception=(e, catch_backtrace())
            end

            try
                atmosphere(
                    T_eff = 5777.0,
                    logg = 4.44,
                    v_mic = 1.0,
                    α_MLT = 1.5,
                    maxiter = 1,
                    eos = eos,
                    opacity = opa_extended,
                    damping = 0.1,
                    τ = 10.0 .^ range(-4.0, 1.0, length=30),
                    use_threads = true,
                    feutrier = true,
                )
                println("✅ Precompiled with ExtendedOpacity.")
            catch e
                @warn "❌ Precompilation M1DIS workload failed with the ExtendedOpacity." exception=(e, catch_backtrace())
            end
        end
        M1DIS.verbose[] = 1
    else
        @warn "Precompilation data not found at $eos_dir. Skipping M1DIS precompilation."
    end
end
