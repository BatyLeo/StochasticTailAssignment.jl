module FlightDelayModel

using ..AircraftRoutingBase
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using LinearAlgebra: dot
using PiecewiseLinearFunctions: PiecewiseLinearFunction
using Random: Random

include("constants.jl")
include("utils.jl")
include("handcrafted_predictor.jl")
include("delay_scenarios.jl")
include("delay_model.jl")
include("delay_propagation.jl")
include("cost.jl")

export DelayCostFunction, IdentityDelayCostFunction
export propagate_delays_from_root_delays, propagate_delays
export delay_expected_cost, delay_expected_total
export full_cost

export softplus, inv_softplus
export DelayCoefficients, DelayEffects, HandcraftedDelayPredictor, weight_table
export departure_prediction, arrival_prediction
export DelayScenarios, nb_scenarios, lognormal_reparameterize, normal_reparameterize
export feature_vector_departure, feature_vector_arrival
export SyntheticDelayModel, has_dynamic_features
export sample_root_scenario, sample_root_scenario_sum
export sample_root_scenarios, sample_root_scenarios_unmerged

end
