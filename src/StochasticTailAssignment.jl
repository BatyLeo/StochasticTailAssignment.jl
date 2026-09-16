module StochasticTailAssignment

using ConstrainedShortestPaths:
    CSPInstance,
    generalized_constrained_shortest_path,
    compute_bounds,
    generalized_a_star,
    topological_order
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs: nv, outneighbors, inneighbors, edges, src, dst
using JuMP
using PiecewiseLinearFunctions:
    PiecewiseLinearFunction, compute_slopes, convex_meet, remove_redundant_breakpoints
using SparseArrays: sparse
using Statistics: mean

include("AircraftRoutingBase/AircraftRoutingBase.jl")
include("FlightDelayModel/FlightDelayModel.jl")
include("InstanceGenerator/InstanceGenerator.jl")

using .AircraftRoutingBase
using .FlightDelayModel
using .InstanceGenerator

include("mip.jl")

include("column_generation/forward_resources.jl")
include("column_generation/forward_extension_functions.jl")
include("column_generation/forward_shortest_path.jl")
include("column_generation/backward_resources.jl")
include("column_generation/backward_extension_functions.jl")
include("column_generation/backward_shortest_path.jl")
include("column_generation/column_generation.jl")
include("column_generation/diving_heuristic.jl")

include("Learning/Learning.jl")

using .Learning

export solve_stochastic_aircraft_routing_mip

export forward_tail_shortest_path
export backward_tail_shortest_path
export stochastic_column_generation, stochastic_column_heuristic
export pure_diving_heuristic, diving_heuristic_with_backtracking!

export compute_features
export generate_dataset, compute_normalization, normalize_data
export build_maximizer, build_loss
export default_model, train_model!
export evaluate_metrics

function plot_gantt end
export plot_gantt

end
