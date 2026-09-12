"""
$TYPEDSIGNATURES

Solve the pricing problem for aircraft `immat_index` and dual variable values `λ_val`.
Uses a resource constrained shortest path enumeration algorithm.
"""
function forward_tail_shortest_path(
    instance::ActivitySchedule,
    immat_index::Int,
    λ_val::AbstractVector,
    root_delays::AbstractMatrix;
    delay_cost_function::PiecewiseLinearFunction,
)
    graph = instance.immat_graphs[immat_index]
    V = nv(instance)
    II = [src(e) for e in edges(graph)]
    JJ = [dst(e) for e in edges(graph)]
    S = size(root_delays, 1)

    origin_vertex = get_s(instance)
    destination_vertex = get_t(instance)

    # Create forward extension functions for each arc
    FF = map(edges(graph)) do e
        u, v = src(e), dst(e)

        # Dual variable associated to u is 0 if not a leg, else it is the value in the vector λ_val
        λ = (is_s(instance, u) || is_maintenance(instance, u)) ? 0.0 : λ_val[u]
        # Operational cost of tail + chaining cost
        cost = edge_cost_origin(instance, u, v, immat_index)

        # Slack between the two activities of the arc
        ω = if is_s(instance, u) || is_t(instance, v)
            Inf
        else
            slack_with_turn_time(get_activity(instance, u), get_activity(instance, v))
        end

        root_delay_scenarios = if is_s(instance, u) || is_maintenance(instance, u)
            zeros(S)
        else
            root_delays[:, u]
        end
        return ForwardExtensionFunction(
            cost,
            λ,
            root_delay_scenarios,
            ω,
            u,
            is_s(instance, u) || is_maintenance(instance, u),
        )
    end

    forward_functions = sparse(II, JJ, FF, V, V)
    starting_resource = TailForwardResource(0.0, 0.0, 0.0, zeros(S))

    csp_instance = CSPInstance(;
        graph,
        origin_vertex,
        destination_vertex,
        origin_forward_resource=starting_resource,
        cost_function=partial_cost,
        forward_functions,
    )
    (; p_star, c_star) = generalized_constrained_shortest_path(
        csp_instance; delay_cost_function
    )
    return (; p_star=p_star[2:(end - 1)], c_star)
end
