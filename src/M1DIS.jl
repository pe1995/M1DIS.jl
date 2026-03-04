module M1DIS

using MUST
using TSO
using TimerOutputs
using DelimitedFiles
using DifferentialEquations
using Interpolations
using Dagger
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays

include("_timing.jl")
include("_feutrier.jl")
include("_constants.jl")
include("_boundary.jl")
include("_hydro.jl")
include("_opacities.jl")
include("_MLT.jl")
include("_atmos.jl")

export atmosphere, save!, activate_timing!, deactivate_timing!, start_timing!, end_timing!

include("precompile.jl")

end
