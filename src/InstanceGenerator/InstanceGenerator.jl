module InstanceGenerator

using Dates
using DocStringExtensions: TYPEDSIGNATURES
using Random
using ..AircraftRoutingBase
using ..FlightDelayModel

include("schedule_internals.jl")
include("delay_sampler.jl")
include("schedule_generator.jl")

export generate_legs, generate_fleet, generate_schedule, generate_root_delays
export generate_benchmark_instance, default_nb_aircraft
export minimum_fleet_size
export build_delay_model

end
