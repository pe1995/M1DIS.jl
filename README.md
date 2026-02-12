# M1DIS.jl
`M1DIS.jl` is a 1D stellar atmosphere julia code that solves the radiative transfer equation and the hydrostatic equilibrium (HE) equation iteratively. `M1DIS.jl` relies on the same microphysics as M3DIS and is thus suitable for the comparison of 1D HE model atmospheres with full 3D RHD M3DIS models.

# Installation
You can install M1DIS.jl simply from the julia REPL:
```julia
using Pkg; Pkg.add("https://github.com/pe1995/M1DIS.jl")
```

# Microphysics
After the intstallation is done, you need to load an opacity table that was generated for `M3DIS`. `M1DIS.jl` is compatible with binned as well as unbinned opacity tables. For this, the `TSO.jl` package is required. Luckily `M1DIS.jl` already contains this package, so you can simply use its functionality 
```julia
using M1DIS

# make sure to load the tables on the T-rho grid!
eos = reload("path/to/eos_table_T.hdf5")
opa_binned = reload("path/to/opacity_table_T.hdf5")

# or alternatively load a un-binned table
opa_unbinned = reload("path/to/unbinned_table.hdf5", mmap=false)

# convert the unbinned table to the format of a binned table.
# This essentially does two things:
#     - kappa => kappa*rho
#     - S => S * dlambda
# which makes the un-binned table usable
# as drop-in replacement for the binned tables,
# which have been converted during the binning process 
M1DIS.TSO.@binned! opa_unbinned eos
```
These tables can then be used to compute opacities and source function at a given temperature and density.

# Computing Atmospheres
To compute a 1D HE atmosphere, simply call the `atmosphere` function and pass the desired effective temperature $T_{\rm eff}$, surface gravity $\log g$, and the optical depth grid you want to compute the model on. Note that the distinction between $\tau_{500}$ and $\tau_{\rm ross}$ is made when you load the EoS table. For $\tau_{500}$, you need to load the corresponding EoS table! Running the code as e.g.
```julia
models = atmosphere(
    T_eff, logg,                         # target Teff and logg
    eos, opacity,                        # tables from above
	τ=10 .^range(-6.0, 2.0, length=100), # optical depth grid
	α_MLT=1.5,                           # Mixing-length parameter
	maxiter=20,                          # maximum number of iterations
	damping=0.1,                         # relative dT step size limit.
	feutrier=true,                       # use the feutrier solver (recommended)
	use_threads=false,                   # use the approximate Feutrier method 
    save_every=1,                        # save every `save_every` snapshot
    dt_tolerance_rel=0.001,              # converged if dT smaller than this
    flux_tolerance_rel=0.001,            # converged if dF smaller than this
	T=nothing, ρ=nothing, P=nothing, z=nothing, # optionally specify starting atmos
) 
```
will iterate 20 times to create an atmosphere. Note that if `save_every=1`, the `models` variable will be an array with 20 entries. If `save_every=-1` only if there is convergence the converged snapshot will be stored. In general you should be able to keep all the variables at the default. 
> **Warning** 
> If you are using a un-binned table, you have to specify use_threads=true! Otherwise the code will use the standard Feutrier solver, which can not handle the resulting gigantic Matrix! Use the approximate solver instead, which will uses a stric tridiagonal matrix and handles frequencies independently. It may be a bit slower in terms of convergence, but it will be much much faster and memory efficient.

# Using the Output
The atmosphere object that is returned is fully compatible with `MUST.jl` and can be used as the output from the `M3DIS` code. For more details see the `MUST.jl` [documentation](https://github.com/pe1995/MUST.jl?tab=readme-ov-file#atmosphere-analysis).