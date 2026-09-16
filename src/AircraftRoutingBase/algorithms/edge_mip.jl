"""
$TYPEDSIGNATURES

Build the MIP model with θ as cost weights in the objective.
"""
function build_model_θ_edge(
    θ::AbstractVector,
    instance::ActivitySchedule;
    model_builder=highs_model,
    silent=true,
    include_chaining_costs=instance.include_chaining_costs,
    use_operational_costs=true,
    relaxation=false,
)
    (; graph, immat_graphs, arc_index, nb_interior_arcs) = instance
    V = nv(graph)
    I = nb_immats(instance)
    s = get_s(instance)
    t = get_t(instance)
    A = length(values(arc_index))

    model = model_builder()
    silent && set_silent(model)

    model[:y] = if relaxation
        @variable(model, y[a in 1:A] >= 0)
    else
        @variable(model, y[a in 1:A], Bin)
    end

    @expression(
        model,
        z⁺[i in 1:I, u in 1:V],
        sum(y[arc_index[u, v, i]] for v in outneighbors(immat_graphs[i], u))
    )

    model[:z] = @expression(
        model,
        z⁻[i in 1:I, v in 1:V],
        sum(y[arc_index[u, v, i]] for u in inneighbors(immat_graphs[i], v))
    )

    @expression(
        model,
        operational_costs,
        sum(
            edge_cost(instance, src(arc), dst(arc), i; include_chaining_costs) *
            y[arc_index[src(arc), dst(arc), i]] for i in 1:I for
            arc in edges(immat_graphs[i])
        )
    )

    @expression(model, delay_costs, sum(θ[ai] * y[ai] for ai in 1:nb_interior_arcs))

    @objective(model, Max, delay_costs - operational_costs * use_operational_costs)

    # flow constraints
    @constraint(model, [i in 1:I], z⁺[i, s] == 1)
    @constraint(model, [i in 1:I], z⁻[i, t] == 1)
    @constraint(model, [v in 1:V, i in 1:I; v != s && v != t], z⁻[i, v] == z⁺[i, v])

    # set covering constraint (all activity must be performed)
    @constraint(model, [v in 1:V; v != s && v != t], sum(z⁻[i, v] for i in 1:I) == 1)

    return model
end

"""
$TYPEDSIGNATURES

Decode routes from the binary variables of the solution of the MIP model.
"""
function decode_routes_from_arc_solution(y_val, instance::ActivitySchedule)
    (; arc_index, immat_graphs) = instance
    arc_index === nothing && error(
        "decode_routes_from_arc_solution requires an ActivitySchedule built with store_arc_index=true",
    )
    I = nb_immats(instance)
    routes = [Route(i, Int[]) for i in 1:I]
    s = get_s(instance)
    t = get_t(instance)

    for i in 1:I
        current_vertex = s
        route = Int[]
        while current_vertex != t
            next_vertex = nothing
            for v in outneighbors(immat_graphs[i], current_vertex)
                if y_val[arc_index[current_vertex, v, i]] > 0.5
                    next_vertex = v
                    break
                end
            end
            next_vertex === nothing && error(
                "decode_routes_from_arc_solution: no successor with y > 0.5 found for " *
                "immat $i at vertex $current_vertex, the arc solution is not a valid route",
            )
            current_vertex = next_vertex
            push!(route, current_vertex)
        end
        routes[i] = Route(i, route[1:(end - 1)])
    end
    return routes
end

"""
$TYPEDSIGNATURES

Recompute binary variables solution of the MIP model from given routes.
"""
function decode_arc_solution_from_routes(
    routes::Vector{Route}, instance::ActivitySchedule; type=Float32
)
    (; arc_index) = instance
    arc_index === nothing && error(
        "decode_arc_solution_from_routes requires an ActivitySchedule built with store_arc_index=true",
    )
    s = get_s(instance)
    t = get_t(instance)
    A = length(values(arc_index))
    y_val = zeros(type, A)
    for route in routes
        current_vertex = s
        i = route.immat_index
        for v in route
            y_val[arc_index[current_vertex, v, i]] = 1
            current_vertex = v
        end
        y_val[arc_index[current_vertex, t, i]] = 1
    end
    return y_val
end

"""
$TYPEDSIGNATURES

CO layer for the aircraft routing problem with edge-based MIP model.
"""
function aircraft_routing_edge_maximizer(
    θ::AbstractVector; instance::ActivitySchedule, verbose=false, kwargs...
)
    model = build_model_θ_edge(θ, instance; kwargs...)
    optimize!(model)
    status = termination_status(model)
    verbose && @info status
    if status != MOI.OPTIMAL
        error(
            "aircraft_routing_edge_maximizer: MIP did not reach an optimal solution (status = $status)",
        )
    end
    y_val = value.(model[:y])
    return y_val
end
