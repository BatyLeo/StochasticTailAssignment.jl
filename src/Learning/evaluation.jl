"""
$TYPEDSIGNATURES

Evaluate `model` on `data` (as produced by [`generate_dataset`](@ref)) by comparing the
routes it induces (through `maximizer`) to the expert routes, in terms of relative
operational cost gap, delay cost gap and full cost gap, along with the average solving
time of `maximizer`.

Costs are computed on `eval_root_delays`, a held-out scenario set distinct from the
`root_delays` used to build the features and expert labels (see
[`generate_dataset`](@ref)), so that the gap reflects out-of-sample quality.

Each gap is only accumulated over the instances where the corresponding expert cost is
strictly positive, and averaged over that same count of instances (not over
`length(data)`), so that instances with a zero expert cost (e.g. no delay cost at all)
do not silently dilute the average. Because of this, the returned gaps are not comparable
to a mean computed over all instances of `data`. `avg_time`, on the other hand, is
averaged over all instances of `data`.
"""
function evaluate_metrics(data, model, maximizer)
    total_op_gap = 0.0
    total_delay_gap = 0.0
    total_cost_gap = 0.0
    total_time = 0.0
    nb_op_gap = 0
    nb_delay_gap = 0
    nb_cost_gap = 0

    for e in data
        (; instance, x, routes, eval_root_delays, delay_cost_function) = e

        y_res = @timed begin
            θ = model(x)
            maximizer(θ; instance)
        end
        y = y_res.value
        total_time += y_res.time

        predicted_routes = decode_routes_from_arc_solution(y, instance)

        predicted_op = operational_cost(predicted_routes, instance)
        expert_op = operational_cost(routes, instance)

        predicted_full = full_cost(
            predicted_routes, eval_root_delays, instance; delay_cost_function
        )
        expert_full = full_cost(routes, eval_root_delays, instance; delay_cost_function)

        predicted_delay = predicted_full - predicted_op
        expert_delay = expert_full - expert_op

        if expert_op > 0
            total_op_gap += (predicted_op - expert_op) / abs(expert_op)
            nb_op_gap += 1
        end
        if expert_delay > 0
            total_delay_gap += (predicted_delay - expert_delay) / abs(expert_delay)
            nb_delay_gap += 1
        end
        if expert_full > 0
            total_cost_gap += (predicted_full - expert_full) / abs(expert_full)
            nb_cost_gap += 1
        end
    end

    n = length(data)
    return (;
        operational_cost_gap=nb_op_gap > 0 ? total_op_gap / nb_op_gap : NaN,
        delay_cost_gap=nb_delay_gap > 0 ? total_delay_gap / nb_delay_gap : NaN,
        full_cost_gap=nb_cost_gap > 0 ? total_cost_gap / nb_cost_gap : NaN,
        avg_time=total_time / n,
    )
end
