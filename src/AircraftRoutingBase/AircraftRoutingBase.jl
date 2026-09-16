module AircraftRoutingBase

using Dates: Dates, @dateformat_str
using DocStringExtensions: SIGNATURES, TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs:
    Graphs,
    ne,
    nv,
    inneighbors,
    outneighbors,
    edges,
    src,
    dst,
    vertices,
    has_edge,
    has_path,
    rem_edge!
using HiGHS: HiGHS
using IterTools: partition
using JSON: JSON
using JuMP  # used stuff should be exported explicitly, but kinda annoying here
using MetaGraphsNext: MetaGraphsNext, code_for, label_for, haskey
using OneHotArrays: onehotbatch, onehot
# using Plots: Plots
using SCIP: SCIP
using SparseArrays: sparse
using Random: Random
using Requires: @require
# using StatsBase: StatsBase, fit, UnitRangeTransform

function __init__()
    @info "If you have Gurobi installed and want to use it, make sure to `using Gurobi` in order to enable it."
    @require Gurobi = "2e9cd046-0924-5485-92f1-d5272153d98b" include("gurobi_setup.jl")
end

include("utils.jl")

include("instance/activity.jl")
include("instance/immat.jl")
include("instance/forced_chainings.jl")
include("instance/activity_schedule.jl")

include("solution/route/abstract_route.jl")
include("solution/route/route.jl")

include("instance/masked_schedule.jl")
include("solution/route/route_with_maintenance.jl")
include("solution/route/partial_route.jl")
include("solution/evaluation.jl")

include("parsing/parsing.jl")

include("algorithms/mip.jl")
include("algorithms/edge_mip.jl")
include("algorithms/colgen.jl")

include("features.jl")

# Basic data types
export AbstractActivity, AbstractLeg, Leg, Maintenance, ForcedChaining, Immat
export slack, slack_with_turn_time
export nb_legs, nb_immats, nb_maintenances, nb_activities
export id, start_time, end_time, start_airport, end_airport, minimum_turn_time, duration
export departure_time, arrival_time, departure_airport, arrival_airport, aircraft_type
export immat,
    is_valid_chaining, flight_distance, departure_airport_size, arrival_airport_size
export start_hour, end_hour, departure_hour, arrival_hour
export get_activity
export is_leg, is_maintenance, is_s, is_t, is_dummy_vertex
export get_last_activity_index

# Instance type, and cost evaluation functions
export AbstractSchedule, ActivitySchedule, MaskedSchedule
export activity_indices, immat_indices, leg_indices
export operational_cost, activity_cost, chaining_cost, edge_cost, edge_cost_origin
export get_s, get_t
export get_maintenance_indices, compute_edge_mask
export get_time_horizon
export compute_arc_index
export mask_schedule, mask_schedule!

export AbstractRoute
export Route, routes_from_routes
export immat_index
export RouteWithMaintenance, RouteWithMaintenance
export leg_indices_from_route
export PartialRoute, route_to_partial_routes

# Parsing functions
export read_instance, write_instance

# Solving and decoding functions
export highs_model, scip_model
export solve_aircraft_routing, decode_routes_from_solution, decode_solution_from_routes
export decode_arc_solution_from_routes, decode_routes_from_arc_solution
export aircraft_routing_edge_maximizer
export column_generation, column_heuristic
export compute_shortest_route

export is_feasible
export string_to_date, to_minutes

export FeaturesConfig,
    compute_leg_features, compute_leg_feature_names, compute_feature_info, schedule_airports

end
