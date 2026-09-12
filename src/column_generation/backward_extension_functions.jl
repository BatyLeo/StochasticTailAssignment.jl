"""
$TYPEDEF

Data structures defining the backward extension functions for the column generation algorithm.
It lives on the arcs of the graph.

# FIELDS
$TYPEDFIELDS
"""
struct TailBackwardExtensionFunction
    "root delay for each scenario of the arc tail"
    root_delays::Vector{Float64}
    "slack between the two activities of the arc"
    slack_with_turn_time::Float64
    "operational cost of the arc tail + chaining cost"
    operational_cost::Float64
    "dual variable λ of the arc tail"
    λ::Float64
end

function change_lambda(f::TailBackwardExtensionFunction, λ::Float64)
    return TailBackwardExtensionFunction(
        f.root_delays, f.slack_with_turn_time, f.operational_cost, λ
    )
end

"""
$TYPEDSIGNATURES

Extend the backward resource `r` with the tail backward extension function `f`.
"""
function (f::TailBackwardExtensionFunction)(
    r::BackwardResource; delay_cost_function::PiecewiseLinearFunction
)
    new_g = map(r.g, f.root_delays) do gⱼ, εⱼ
        delay_propagation_function = if f.slack_with_turn_time == Inf
            PiecewiseLinearFunction([0.0], [0.0], 0.0, 0.0)
        else
            PiecewiseLinearFunction([0.0], [εⱼ - f.slack_with_turn_time], 0.0, 1.0)
        end
        arrival_delay_function = PiecewiseLinearFunction([0.0], [εⱼ], 1.0, 1.0)
        return remove_redundant_breakpoints(
            delay_cost_function ∘ arrival_delay_function + gⱼ ∘ delay_propagation_function;
            atol=1e-8,
        )
    end
    new_λ = r.λ_sum .+ f.λ
    new_operational_cost_sum = r.operational_cost_sum + f.operational_cost
    return BackwardResource(new_g, new_operational_cost_sum, new_λ)
end

"""
$TYPEDSIGNATURES

Extend the backward resource `r` with the tail backward extension function `f`.
"""
function (f::TailBackwardExtensionFunction)(r::PartialBackwardResource; kwargs...)
    new_λ = r.λ_sum .+ f.λ
    new_operational_cost_sum = r.operational_cost_sum + f.operational_cost
    return PartialBackwardResource(new_operational_cost_sum, new_λ)
end
