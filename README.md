# M1DIS.jl
`M1DIS.jl` is a 1D stellar atmosphere julia code that solves the radiative transfer equation and the hydrostatic equilibrium (HE) equation iteratively. `M1DIS.jl` relies on the same microphysics as M3DIS and is thus suitable for the comparison of 1D HE model atmospheres with full 3D RHD M3DIS models.

# Installation
You can install M1DIS.jl simply from the julia REPL:
```julia
using Pkg
Pkg.add(url="https://github.com/pe1995/MUST.jl.git")
Pkg.add(url="https://github.com/pe1995/TSO.jl.git")
Pkg.add(url="https://github.com/pe1995/M1DIS.jl")
```

Or clone this repository and just do `Pkg.instantiate()` within. If you want to use the command line tools, I recommend cloning instead of installing through the REPL.
```bash
git clone https://github.com/pe1995/M1DIS.jl.git
julia --project="./M1DIS.jl" -e 'using Pkg; Pkg.instantiate()'
```

# Microphysics
After the intstallation is done, you need to load an opacity table that was generated for `M3DIS`. `M1DIS.jl` is compatible with binned as well as unbinned opacity tables. For this, the `TSO.jl` package is required. Luckily `M1DIS.jl` already contains this package, so you can simply use its functionality 
```julia
using M1DIS
using TSO

# make sure to load the tables on the T-rho grid!
eos = reload("path/to/eos_table_T.hdf5")
opa_binned = reload("path/to/opacity_table_T.hdf5")) |> extended

# or alternatively load a un-binned table
opa_unbinned = reload("path/to/unbinned_table.hdf5", mmap=true) |> extended

# you can also ignore the source function for un-binned tables by using TSO.MiniOpacityTable
opa_unbinned_mini = reload(TSO.MiniOpacityTable, "path/to/unbinned_table.hdf5")
```
These tables can then be used to compute opacities and source function at a given temperature and density. The code should detect automaticaly if you are passing an unbinned table. If for some reason this is not the case, you can skip the convenient `extended()` function and use `TSO.ExtendedOpacity(opa=TSO.reload(opa_file, mmap=true), binned=false)` instead.

# Computing Atmospheres
To compute a 1D HE atmosphere, simply call the `atmosphere` function and pass the desired effective temperature $T_{\rm eff}$, surface gravity $\log g$, and the optical depth grid you want to compute the model on. Note that the distinction between $\tau_{500}$ and $\tau_{\rm ross}$ is made when you load the EoS table. For $\tau_{500}$, you need to load the corresponding EoS table! Running the code as e.g.
```julia
models = atmosphere(
    T_eff, logg,                         # target Teff and logg
    eos, opacity,                        # tables from above
	τ=10 .^range(-6.0, 2.0, length=100), # optical depth grid
	α_MLT=1.5,                           # Mixing-length parameter
	maxiter=50,                          # maximum number of iterations
	damping=0.1,                         # relative dT step size limit.
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
The atmosphere object that is returned is fully compatible with `MUST.jl` and can be used as the output from the `M3DIS` code. For more details see the `MUST.jl` [documentation](https://github.com/pe1995/MUST.jl?tab=readme-ov-file#atmosphere-analysis). You can save the result in Multi1D or Multi3D format by using the `save!` function.
```julia
save!(
	models[end], "my_model_name"; 
	folder = out_dir, 
	vmic = vmic,     # added to the output atmosphere 
	logg = logg,
	eos500 = eos500	 # For Multi1D a tau500 scale is needed. You can pass the EoS here.
)
```

# Command line tools
In case you want to run `M1DIS.jl` from the command line directly, there are scripts available in the `bin/` directory. For star and planet mode, there are default input files available that you can use to run the code. Note that you can override the defaults in the file via the command line directly. This means something like the following will work,
```bash
julia -t 10 bin/m1dis_star.jl -c bin/star_config.toml --teff=6500 --logg=4.0 --maxiter=100
```
and will create a stellar atmosphere (using 10 CPU threads) with all the default parameters selected in `bin/star_config.toml`, but with an effective temperature of 6500K and surface gravity of 4.0. The code will stop after 100 iterations and store each iteration for you to explore. For all available parameters see `julia bin/m1dis_star.jl --help`. There also is a version available specifically for planets. 
Note that if you do not specify an EoS, the code will automatically generate an EoS table using Multi3D. For this, please make sure that the settings in `bin/eos_config.toml` are correct. You can then just pass the chemical composition as arguments and the corresponding table will be created. These tables can be quite large, so make sure you have enough RAM and disc space for this computation.

Besides creating the models, the command line tool can also be used to compute the corresponding spectra. For this, there are example scripts available in `bin/spectrum.jl`. The principle is the same:
```bash
julia bin/spectrum.jl -c bin/spectrum_config.toml -m models/m1dis_model -n 'myspectrum' --feh=-1 --alpha=0.4
```
Where the default parameters are again stored in a `.toml` file. Multi3D is assumed to be located at the path you specified in the `bin/eos_config.toml`.

# Opacity Binning
There is an additional usecase of `M1DIS.jl`, which provides great synergy with the 3D RHD code `M3DIS`. In order to start a 3D simulations, there a two very important things that need to be present: The initial atmospheric structure and the opacity table. Because `M3DIS` works with binned opacities, there is a complementary code that computes these binned tables already available (`TSO.jl`, see above). However, there is generally a great uncertainty associated with the bin selection, which is why it is advisable to automate the procedure. `M1DIS.jl` is capable of doing this in a very straight forward way. 

For this, the command line tool `bin/opacity_binning.jl` with the default configuration `bin/binning_config.toml` is available. This script computes -- for a given chemical compostion -- the monochromatic opacity table using `Multi3D`. After specifing $T_{eff}$, $\log g$ the 1D HE atmosphere will be computed. After the atmosphere has been constructed, the opacity binning is optimized iteratively by adjusting the bin assignment matrix with dimensions ($n_{\lambda} \times n_{bins}$), computing the respective binned opacity, and solving the RT using those opacities. The goodness of the binning is judged by comparing cooling error at the cooling-peak with the unbinned result. The code furthermore makes sure that the flux error is kept within bounds.

