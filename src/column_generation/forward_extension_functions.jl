"""
$TYPEDEF

Data structures defining the forward extension functions for the column generation algorithm.
It lives on the arcs of the graph.
It is used to expand resources at the tail of arcs (see [`TailForwardResource`](@ref)).
The considered node index of the arc is stored in the field `activity_index`.

# Fields
$TYPEDFIELDS
"""
struct ForwardExtensionFunction
    "leg cost of node + chaining cost"
    operational_cost::Float64
    "dual variable λ of node"
    λ::Float64
    "root delay of node for each scenario"
    root_delays::Vector{Float64}
    "slack (with turn time) between the two activities of the arc"
    slack_with_turn_time::Float64
    "activity index of node"
    activity_index::Int
    "true if tail of arc is a maintenance activity (or dummy), used for knowing when not to compute delay costs"
    is_tail_maintenance::Bool
end

function change_lambda(f::ForwardExtensionFunction, λ::Float64)
    return ForwardExtensionFunction(
        f.operational_cost,
        λ,
        f.root_delays,
        f.slack_with_turn_time,
        f.activity_index,
        f.is_tail_maintenance,
    )
end

"""
$TYPEDSIGNATURES

Extend the forward resource `r` with the tail forward extension function `f`.
"""
function (f::ForwardExtensionFunction)(
    r::TailForwardResource; delay_cost_function::PiecewiseLinearFunction
)
    # extend resource
    new_operational_cost_sum = r.operational_cost_sum + f.operational_cost
    new_λ_sum = r.λ_sum + f.λ

    new_ξ = f.root_delays .+ r.propagated_delay

    # compute delay costs only on legs (no cost on maintenance activities, only propagation)
    new_delay_cost_sum =
        r.delay_cost_sum +
        (f.is_tail_maintenance ? 0.0 : mean(delay_cost_function(e) for e in new_ξ))

    new_ξ .= max.(new_ξ .- f.slack_with_turn_time, 0.0)

    return TailForwardResource(
        new_operational_cost_sum, new_delay_cost_sum, new_λ_sum, new_ξ
    ),
    true # always feasible
end
