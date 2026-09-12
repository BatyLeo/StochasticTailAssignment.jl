"""
$TYPEDSIGNATURES

Compute a shortest path between `start_node` and `end_node`, with weight matrix `weights`.
Uses the ford bellman algorithm, since the graph is acyclic.
"""
function compute_shortest_subpath(instance::ActivitySchedule, start_node, end_node, weights)
    @assert start_node != end_node "start $start_node and end node $end_node must be different"
    p = Graphs.bellman_ford_shortest_paths(instance.graph, start_node, weights)
    path = [end_node]
    while true
        prev = p.parents[path[1]]
        if prev == start_node
            break
        end
        pushfirst!(path, prev)
    end
    return path
end

"""
$TYPEDSIGNATURES

Solve the pricing problem for given aircraft `immat_index` and dual variables `λ_val`.
This is solved as successive shortest paths between scheduled maintenances for `immat_index`.
"""
function compute_shortest_route(
    instance::ActivitySchedule, immat_index::Int, λ_val; include_chaining_costs, silent
)
    (; maintenances) = instance

    L = nb_legs(instance)
    M = nb_maintenances(instance)
    V = nv(instance)

    s = get_s(instance)
    t = get_t(instance)

    II = [src(e) for e in edges(instance)]
    JJ = [dst(e) for e in edges(instance)]
    K = [
        edge_cost(instance, src(e), dst(e), immat_index; include_chaining_costs) -
        (dst(e) <= L ? λ_val[dst(e)] : 0.0) +
        ((src(e) == s && dst(e) == t) ? 1000000.0 : 0.0) for e in edges(instance)
    ]  # TODO: cleanup
    @assert length(λ_val) == L
    weights = sparse(II, JJ, K, V, V)

    maintenance_indices = [
        m for m in 1:M if maintenances[m].immat == instance.immats[immat_index].id
    ]
    immat_maintenances = maintenances[maintenance_indices]
    maintenance_indices =
        maintenance_indices[sortperm(immat_maintenances; by=x -> x.start_time)] .+ L

    # start path at last activity, if there is one
    path_start = get_last_activity_index(instance, immat_index)
    # if last activity is a maintenance, remove it
    if length(maintenance_indices) > 0 && path_start == maintenance_indices[1]
        maintenance_indices = maintenance_indices[2:end]
    end

    current_node = path_start

    final_path = path_start == get_s(instance) ? Int[] : [path_start]

    for m in maintenance_indices
        path = compute_shortest_subpath(instance, current_node, m, weights)
        # remove maintenance from other aircrafts (should not affect the cost)
        mask = [!is_maintenance(instance, e) for e in path[1:(end - 1)]]
        path = path[vcat(mask, true)]
        # if any([is_maintenance(instance, e) for e in path[1:(end - 1)]])
        #     @info [is_maintenance(instance, e) for e in path]
        #     silent || @warn "path contains maintenance"
        # end # TODO: remove these additional maintenances to keep feasibility (should not affect cost)
        final_path = vcat(final_path, path)
        current_node = final_path[end] # = m
        @assert current_node == m
    end
    path = compute_shortest_subpath(instance, current_node, get_t(instance), weights)
    final_path = vcat(final_path, path[1:(end - 1)])

    # TODO: fix route costs (missing s - last activity cost)
    shortest_route = Route(immat_index, final_path)
    is_feasible(shortest_route, instance)
    return shortest_route #, 0.0
end

"""
$TYPEDSIGNATURES

Perform column generation to solve the deterministic tail assignment problem.

Returns:
- `columns`: vector of generated columns as a `Vector{Route}`
- `model`: final JuMP model
- `VV`: reduced cost evolution for each aircraft
"""
function column_generation(
    instance::ActivitySchedule,
    initial_paths::Vector{Route};
    model_builder=highs_model,
    include_chaining_costs=instance.include_chaining_costs,
    max_iterations=1000,
    tol=1e-8,
    silent=false,
)
    I = nb_immats(instance)

    model = model_builder()
    set_silent(model)

    L = nb_legs(instance)
    I = nb_immats(instance)

    @variable(model, λ[l in 1:L])
    @variable(model, μ[i in 1:I] <= 0)

    @objective(model, Max, sum(λ[l] for l in 1:L) + sum(μ[i] for i in 1:I))

    for route in initial_paths
        i = route.immat_index
        cp = operational_cost(route, instance; include_chaining_costs)
        Lp = [l for l in route if l <= L]
        @constraint(model, cp - sum(λ[l] for l in Lp) - μ[i] >= 0)
    end
    VV = [Float64[] for i in 1:I]
    done = falses(I)
    columns = deepcopy(initial_paths)

    for it in 1:max_iterations
        if all(done)
            silent || @info "Solution found at iteration $it"
            break
        end
        done .= false

        optimize!(model)
        λ_val = value.(λ)
        μ_val = value.(μ)

        for i in 1:I
            shortest_route = compute_shortest_route(
                instance, i, λ_val; include_chaining_costs, silent
            )
            push!(columns, shortest_route)

            cp = operational_cost(shortest_route, instance; include_chaining_costs)
            Lp = [l for l in shortest_route if l <= L]

            vv = cp - sum(λ_val[l] for l in Lp; init=zero(eltype(λ_val))) - μ_val[i]
            push!(VV[i], vv)

            if vv >= -tol
                done[i] = true
                continue
            end

            @constraint(model, cp - sum(λ[l] for l in Lp) - μ[i] >= 0)
        end
    end

    optimize!(model)
    return columns, model, VV
end

"""
$TYPEDSIGNATURES

Take a vector of `Route`s and returns the solution of the restricted master problem heuristic for the deterministic tail assignment problem.
"""
function column_heuristic(
    instance::ActivitySchedule,
    columns::Vector{Route};
    model_builder=highs_model,
    include_chaining_costs=instance.include_chaining_costs,
    silent=false,
    bin=true,
    column_cost=p -> operational_cost(p, instance; include_chaining_costs),
    time_limit=nothing,
    warm_start=false,
)
    I = nb_immats(instance)
    LL = nb_activities(instance)

    C = [[p for p in columns if p.immat_index == i] for i in 1:I]

    model = model_builder()
    silent && set_silent(model)

    isnothing(time_limit) || set_time_limit_sec(model, time_limit)

    start_f(i, p) = warm_start ? p in columns[1:I] : nothing
    if bin
        @variable(model, y[i in 1:I, p in C[i]], Bin, start = start_f(i, p))
    else
        @variable(model, y[i in 1:I, p in C[i]] >= 0, start = start_f(i, p))
    end

    @objective(model, Min, sum(column_cost(p) * y[i, p] for i in 1:I for p in C[i]))

    # Set partition constraint
    @constraint(
        model,
        lambda[l in 1:LL],
        sum(y[i, p] for i in 1:I for p in C[i] if l in p) == 1  # && i == p.immat_index
    )

    # one path per immat
    @constraint(model, mu[i in 1:I], sum(y[i, p] for p in C[i]) <= 1)

    optimize!(model)

    y_val = value.(y)
    res = Route[]
    for i in 1:I
        for p in C[i]
            if y_val[i, p] > 0.99
                push!(res, p)
            end
        end
    end

    return res, objective_value(model)
end
