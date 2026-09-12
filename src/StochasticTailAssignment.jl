module StochasticTailAssignment

include("AircraftRoutingBase/AircraftRoutingBase.jl")
include("FlightDelayModel/FlightDelayModel.jl")
include("InstanceGenerator/InstanceGenerator.jl")

using .AircraftRoutingBase
using .FlightDelayModel
using .InstanceGenerator

using DocStringExtensions: TYPEDSIGNATURES
using Graphs: nv, outneighbors, inneighbors
using JuMP
using PiecewiseLinearFunctions: PiecewiseLinearFunction, compute_slopes

include("mip.jl")

export solve_stochastic_aircraft_routing_mip

end
