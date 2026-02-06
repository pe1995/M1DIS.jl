module M1DIS

include("FeutrierRT.jl")

using .FeutrierRT
using MUST
using TSO
using DelimitedFiles
using DifferentialEquations
using Interpolations
using Dagger
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays

include("_constants.jl")
include("_boundary.jl")
include("_hydro.jl")
include("_feutrier.jl")
include("_RT.jl")
include("_MLT.jl")
include("_atmos.jl")

export atmosphere

end
