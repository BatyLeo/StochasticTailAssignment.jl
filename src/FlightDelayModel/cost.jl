"""
$TYPEDSIGNATURES

Build a delay cost function from breakpoints and slopes, as a piecewise linear function.
"""
function DelayCostFunction(;
    breakpoints::Vector{T}=DELAY_COST_BREAKPOINTS, slopes::Vector{T}
) where {T}
    @assert length(breakpoints) == length(slopes) - 1
    new_y = T[0.0]
    new_x = T[0.0]
    for (s, x) in zip(slopes, breakpoints)
        y = new_y[end] + s * (x - new_x[end])
        push!(new_y, y)
        push!(new_x, x)
    end
    return PiecewiseLinearFunction(new_x, new_y, 0.0, slopes[end])
end

"""
Construct an identity delay cost function where the cost is equal to the delay.
"""
function IdentityDelayCostFunction()
    return PiecewiseLinearFunction([0.0], [0.0], 0.0, 1.0)
end

function delay_expected_cost(
    route_s::Union{Route,Vector{Route}},
    root_delays::AbstractMatrix,
    instance::ActivitySchedule;
    delay_cost_function,
)
    arrival_delays = propagate_delays_from_root_delays(route_s, root_delays, instance)
    S = size(root_delays, 1)
    return sum(delay_cost_function(ξ) for ξ in arrival_delays) / S
end

function delay_expected_cost(
    route_s::Union{Route,Vector{Route}},
    root_delays::AbstractMatrix,
    instance::MaskedSchedule;
    delay_cost_function,
)
    return delay_expected_cost(route_s, root_delays, instance.schedule; delay_cost_function)
end

function delay_expected_total(
    route_s, root_delays::AbstractMatrix, instance::AbstractSchedule
)
    return delay_expected_cost(
        route_s, root_delays, instance; delay_cost_function=IdentityDelayCostFunction()
    )
end

function full_cost(
    route_s::Union{Route,Vector{Route}},
    root_delays::AbstractMatrix,
    instance::AbstractSchedule;
    delay_cost_function,
)
    delay_cost = delay_expected_cost(route_s, root_delays, instance; delay_cost_function)
    return operational_cost(route_s, instance) + delay_cost
end
