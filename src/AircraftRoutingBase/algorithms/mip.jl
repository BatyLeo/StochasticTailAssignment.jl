"""
$TYPEDSIGNATURES

Initialize a HiGHS model.
"""
function highs_model()
    model = Model(HiGHS.Optimizer)
    return model
end

"""
$TYPEDSIGNATURES

Initialize a SCIP model.
"""
function scip_model()
    model = Model(SCIP.Optimizer)
    return model
end

"""
$TYPEDSIGNATURES

Build the MIP model.
"""
function build_model(
    instance::AbstractSchedule;
    model_builder=highs_model,
    include_chaining_costs=does_include_chaining_costs(instance),
    silent=false,
    feasibility_only=false,
    relaxation=false,
)
    graph = instance.graph
    immat_graphs = instance.immat_graphs
    s = get_s(instance)
    t = get_t(instance)
    V = nv(graph)
    I = immat_indices(instance) # nb_immats(instance)

    model = model_builder()
    silent && set_silent(model)

    if relaxation
        model[:y] = @variable(
            model, y[i in I, u in 1:V, v in outneighbors(immat_graphs[i], u)] >= 0
        )
    else
        model[:y] = @variable(
            model, y[i in I, u in 1:V, v in outneighbors(immat_graphs[i], u)], Bin
        )
    end
    @expression(
        model,
        z⁺[u in 1:V, i in I],
        sum(y[i, u, v] for v in outneighbors(immat_graphs[i], u))
    )

    @expression(
        model,
        z⁻[v in 1:V, i in I],
        sum(y[i, u, v] for u in inneighbors(immat_graphs[i], v))
    )

    @expression(
        model,
        operational_costs,
        sum(
            edge_cost(instance, u, v, i; include_chaining_costs) * y[i, u, v] for i in I for
            u in 1:V for v in outneighbors(immat_graphs[i], u)
        )
    )

    if feasibility_only
        @objective(model, FEASIBILITY_SENSE, 0.0)
    else
        @objective(model, Min, operational_costs)
    end

    # flow constraints
    @constraint(model, [i in I], z⁺[s, i] == 1)
    @constraint(model, [i in I], z⁻[t, i] == 1)
    @constraint(model, [v in 1:V, i in I; v != s && v != t], z⁻[v, i] == z⁺[v, i])

    # set covering constraint (all legs must be performed, maintenances are ensured by graph preprocessing)
    L = leg_indices(instance)
    @constraint(model, [v in L; v != s && v != t], sum(z⁻[v, i] for i in I) == 1)

    return model
end

"""
$TYPEDSIGNATURES

Decode routes from the binary variables of the solution of the MIP model.
"""
function decode_routes_from_solution(y_val, instance::AbstractSchedule)
    I = immat_indices(instance) # nb_immats(instance)
    routes = Route[]
    s = get_s(instance)
    t = get_t(instance)

    for i in I
        current_vertex = s
        route = Int[]
        while current_vertex != t
            for v in outneighbors(instance.immat_graphs[i], current_vertex)
                if y_val[i, current_vertex, v] > 0.5
                    current_vertex = v
                    push!(route, v)
                    break
                end
            end
        end
        push!(routes, Route(i, route[1:(end - 1)]))
    end
    return routes
end

"""
$TYPEDSIGNATURES

Recompute binary variables solution of the MIP model from given routes.
"""
function decode_solution_from_routes(routes, instance)
    I = nb_immats(instance)
    s = get_s(instance)
    t = get_t(instance)
    V = nv(instance)

    y_val = zeros(V, V, I)
    for (i, route) in enumerate(routes)
        current_vertex = s
        for v in route
            y_val[i, current_vertex, v] = 1
            current_vertex = v
        end
        y_val[i, current_vertex, t] = 1
    end
    return y_val
end

"""
$TYPEDSIGNATURES

Solve the aircraft routing problem using a MIP solver.
Returns the optimal routes and objective value.
"""
function solve_aircraft_routing(
    instance::AbstractSchedule; relaxation=false, silent=true, kwargs...
)
    model = build_model(instance; silent, relaxation, kwargs...)
    optimize!(model)
    silent || @info termination_status(model)
    if termination_status(model) != MOI.OPTIMAL
        return Vector{Int}[], -1, nothing
    end
    y_val = value.(model[:y])
    if relaxation
        return nothing, objective_value(model), y_val
    end
    return decode_routes_from_solution(y_val, instance), objective_value(model), y_val
end
