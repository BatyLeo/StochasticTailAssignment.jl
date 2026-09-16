module Learning

using DocStringExtensions: TYPEDSIGNATURES
using Statistics: quantile, mean
using Graphs: edges, src, dst
using Flux: Flux, Chain, Dense, relu, gradient, Adam
using InferOpt: LinearMaximizer, PerturbedAdditive, FenchelYoungLoss
using ..AircraftRoutingBase
using ..FlightDelayModel: full_cost, DelayScenarios, sample_root_scenarios_unmerged
using ..InstanceGenerator:
    generate_benchmark_instance, generate_root_delays, build_delay_model
import ..diving_heuristic_with_backtracking!
import ..stochastic_column_generation

include("features.jl")
include("dataset.jl")
include("pipeline.jl")
include("training.jl")
include("evaluation.jl")

export compute_features
export generate_dataset, compute_normalization, normalize_data
export build_maximizer, build_loss
export default_model, train_model!
export evaluate_metrics

end
