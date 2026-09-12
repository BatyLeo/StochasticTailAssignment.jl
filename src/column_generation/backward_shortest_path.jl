function merge_bounds!(current_bounds, partial_bounds, new_bounds)
    # bounds = deepcopy(partial_bounds)
    for k in keys(partial_bounds)
        current_bounds[k] = BackwardResource(partial_bounds[k], new_bounds[k])
    end
    return current_bounds
end

"""
$TYPEDSIGNATURES

Precompute partial bounds for aircraft `immat_index`.
"""
function compute_partial_bounds(
    instance::AbstractSchedule,
    root_delays::AbstractMatrix,
    immat_index::Int;
    delay_cost_function::PiecewiseLinearFunction,
)
    graph = instance.immat_graphs[immat_index]
    V = nv(instance)
    II = [src(e) for e in edges(graph)]
    JJ = [dst(e) for e in edges(graph)]
    S = size(root_delays, 1)

    origin_vertex = get_s(instance)
    destination_vertex = get_t(instance)

    BF = map(edges(graph)) do e
        u, v = src(e), dst(e)
        λ = 0.0

        ε = if is_s(instance, u) || is_maintenance(instance, u)
            zeros(S)
        else
            root_delays[:, u]
        end

        # Slack between the two activities of the arc
        ω = if is_s(instance, u) || is_t(instance, v)
            Inf
        else
            slack_with_turn_time(get_activity(instance, u), get_activity(instance, v))
        end

        cost = edge_cost_origin(instance, u, v, immat_index)

        return TailBackwardExtensionFunction(ε, ω, cost, λ)
    end
    backward_functions = sparse(II, JJ, BF, V, V)

    destination_resource = BackwardResource(
        [PiecewiseLinearFunction([0.0], [0.0], 0.0, 0.0) for _ in 1:S], 0.0, 0.0
    )

    csp_instance = CSPInstance(;
        graph,
        origin_vertex,
        destination_vertex,
        origin_forward_resource=nothing,
        destination_backward_resource=destination_resource,
        cost_function=nothing,
        forward_functions=zeros(1, 1),
        backward_functions,
    )
    return compute_bounds(csp_instance; delay_cost_function)
end

"""
$TYPEDSIGNATURES

Precompute partial bounds for each aircraft.
"""
function compute_partial_bounds(
    instance::AbstractSchedule,
    root_delays::AbstractMatrix;
    delay_cost_function::PiecewiseLinearFunction,
)
    return [
        compute_partial_bounds(instance, root_delays, i; delay_cost_function) for
        i in immat_indices(instance)
    ]
end

"""
$TYPEDSIGNATURES

Solve the pricing problem for aircraft `immat_index` and dual variable values `λ_val`.
Uses a resource constrained shortest path enumeration algorithm with bounding.
"""
function backward_tail_shortest_path(
    instance::AbstractSchedule,
    immat_index::Int,
    λ_val::AbstractVector,
    root_delays::AbstractMatrix,
    partial_bounds,
    current_bounds,
    FF,
    BF,
    order;
    delay_cost_function::PiecewiseLinearFunction,
)
    graph = instance.immat_graphs[immat_index]
    V = nv(instance)
    II = [src(e) for e in edges(graph)]
    JJ = [dst(e) for e in edges(graph)]
    S = size(root_delays, 1)

    origin_vertex = get_s(instance)
    destination_vertex = get_t(instance)

    for (edge_index, e) in enumerate(edges(graph))
        u = src(e)
        # Dual variable associated to u is 0 if not a leg, else it is the value in the vector λ_val
        λ = (is_s(instance, u) || is_maintenance(instance, u)) ? 0.0 : λ_val[u]
        FF[edge_index] = change_lambda(FF[edge_index], λ)
        BF[edge_index] = change_lambda(BF[edge_index], λ)
    end

    backward_functions = sparse(II, JJ, BF, V, V)
    forward_functions = sparse(II, JJ, FF, V, V)
    starting_resource = TailForwardResource(0.0, 0.0, 0.0, zeros(S))

    destination_resource = PartialBackwardResource(0.0, 0.0)

    csp_instance = CSPInstance(;
        graph,
        origin_vertex,
        destination_vertex,
        origin_forward_resource=starting_resource,
        destination_backward_resource=destination_resource,
        cost_function=stochastic_cost,
        forward_functions,
        backward_functions,
        topological_ordering=order,
    )
    new_bounds = compute_bounds(csp_instance; delay_cost_function)
    bounds = merge_bounds!(current_bounds, partial_bounds, new_bounds)
    (; p_star, c_star, info) = generalized_a_star(csp_instance, bounds; delay_cost_function)
    # remove s and t from the path
    return (; p_star=p_star[2:(end - 1)], c_star)
end
