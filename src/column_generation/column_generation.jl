function _constrained_shortest_path(;
    backward,
    instance,
    i,
    λ_val,
    root_delays,
    delay_cost_function,
    partial_bound,
    current_bound,
    FF,
    BF,
    order,
)
    return shortest_route, route_cost = if !backward
        forward_tail_shortest_path(instance, i, λ_val, root_delays; delay_cost_function)
    else
        backward_tail_shortest_path(
            instance,
            i,
            λ_val,
            root_delays,
            partial_bound,
            current_bound,
            FF,
            BF,
            order;
            delay_cost_function,
        )
    end
end

# master problem
function _solve_master_problem!(model, λ, μ)
    optimize!(model)
    if is_solved_and_feasible(model)
        λ_val = value.(λ)
        μ_val = value.(μ)
        return λ_val, μ_val, objective_value(model), true
    end
    # else
    return nothing, nothing, nothing, false
end

# pricing problem for immat i
function _solve_subproblem!(;
    model,
    columns,
    cuts,
    VV,
    instance,
    tol,
    silent,
    it,
    i,
    λ,
    μ,
    root_delays,
    delay_cost_function,
    backward,
    partial_bound,
    current_bound,
    λ_val,
    μ_val,
    FF,
    BF,
    order,
)
    shortest_route, route_cost = _constrained_shortest_path(;
        backward=backward,
        instance=instance,
        i=i,
        λ_val=λ_val,
        root_delays=root_delays,
        delay_cost_function=delay_cost_function,
        partial_bound=partial_bound,
        current_bound=current_bound,
        FF=FF,
        BF=BF,
        order,
    )

    Lp = leg_indices_from_route(shortest_route, instance)
    λ_sum = sum(λ_val[l] for l in Lp; init=zero(eltype(λ_val)))
    cp = route_cost + λ_sum
    # Total path cost
    vv = route_cost - μ_val[i]
    push!(VV[i], vv)
    silent || @info "$it | $i | $vv | $(μ_val[i]) | $cp | $λ_sum"

    if vv >= -tol
        return false, vv, route_cost
    end
    # Add a cut, reduced cost is negative
    push!(columns, Route(i, shortest_route))
    push!(cuts, @constraint(model, cp - sum(λ[l] for l in Lp) - μ[i] >= 0))
    return true, vv, route_cost
end

function _precompute_extension_functions(instance, immat_index, root_delays)
    graph = instance.immat_graphs[immat_index]
    S = size(root_delays, 1)

    # Create forward extension functions for each arc
    FF = map(edges(graph)) do e
        u, v = src(e), dst(e)

        λ = 0.0  # placeholder

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

    BF = map(edges(graph)) do e
        u, v = src(e), dst(e)
        λ = 0.0  # placeholder

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

    return (; FF, BF)
end

function _precompute_topological_orders(instance::AbstractSchedule)
    I = immat_indices(instance)
    s = get_s(instance)
    t = get_t(instance)
    return map(I) do i
        return topological_order(instance.immat_graphs[i], s, t)
    end
end

"""
$TYPEDSIGNATURES

Perform column generation to solve the stochastic tail assignment problem.

# Arguments
- `instance`: an instance of `AbstractSchedule` (also works with `MaskedSchedule`)
- `initial_paths`: initial paths to start the column generation, should at least include one path per aircraft
- `model_builder`: function to build the JuMP model
- `root_delays`: root_delays[s, l] is the delay of leg l in scenario s
- `delay_cost_function`: function mapping delays to costs
- `max_nb_columns`: maximum number of columns to generate
- `tol`: tolerance for reduced cost
- `silent`: whether to print information

# Returns:
- `columns`: vector of generated columns as a `Vector{Route}`
- `model`: final JuMP model
- `VV`: reduced cost evolution for each aircraft
"""
function stochastic_column_generation(
    instance::AbstractSchedule,
    initial_paths::Vector{Route};
    model_builder=highs_model,
    root_delays,
    delay_cost_function,
    max_nb_columns=1000,
    tol=1e-8,
    silent=true,
    backward=true,
    add_all_columns=true,
    starting_sweep=true,
)
    model = model_builder()
    set_silent(model)

    L = leg_indices(instance) # nb_legs(instance)
    I = immat_indices(instance) # nb_immats(instance)

    @variable(model, λ[l in L])
    @variable(model, μ[i in I])

    @objective(model, Max, sum(λ[l] for l in L) + sum(μ[i] for i in I))

    cuts = map(initial_paths) do route
        cp = full_cost(route, root_delays, instance; delay_cost_function)
        Lp = leg_indices_from_route(route, instance)
        i = immat_index(route)
        return @constraint(model, cp - sum(λ[l] for l in Lp) - μ[i] >= 0)
    end

    VV = Dict([i => Float64[] for i in I])
    ub_history = Float64[]
    lb_history = Float64[]
    columns = deepcopy(initial_paths)

    partial_bounds = compute_partial_bounds(instance, root_delays; delay_cost_function)
    topological_orders = _precompute_topological_orders(instance)
    precomputed_extension_functions = [
        _precompute_extension_functions(instance, immat_index, root_delays) for
        immat_index in I
    ]
    current_bounds = deepcopy(partial_bounds)

    silent || @info "Starting column generation"

    lb = -Inf

    it = 0
    # Starting sweep
    if starting_sweep
        for (ii, i) in enumerate(I)
            while true
                it += 1
                λ_val, μ_val, _, is_good = _solve_master_problem!(model, λ, μ)
                if !is_good
                    return (; feasible=false)
                end
                (; FF, BF) = something(precomputed_extension_functions)[ii]
                partial_bound = something(partial_bounds)[ii]
                current_bound = something(current_bounds)[ii]
                order = something(topological_orders)[ii]
                cut_was_added, _, _ = _solve_subproblem!(;
                    model,
                    columns,
                    cuts,
                    VV,
                    instance,
                    tol,
                    silent,
                    it,
                    i,
                    λ,
                    μ,
                    root_delays,
                    delay_cost_function,
                    backward,
                    partial_bound,
                    current_bound,
                    λ_val,
                    μ_val,
                    FF,
                    BF,
                    order,
                )
                cut_was_added || break
            end
        end
    end

    done = false
    while length(columns) < max_nb_columns
        if done
            silent || @info "Solution found at iteration $it"
            break
        end
        λ_val, μ_val, ub, is_good = _solve_master_problem!(model, λ, μ)
        if !is_good
            return (; feasible=false)
        end

        # Solve a constrained shortest path problem for each aircraft, and add a cut if needed
        done = true
        cp_sum = 0.0
        for (ii, i) in enumerate(I)
            it += 1
            (; FF, BF) = something(precomputed_extension_functions)[ii]
            partial_bound = something(partial_bounds)[ii]
            current_bound = something(current_bounds)[ii]
            order = something(topological_orders)[ii]
            cut_was_added, _, cp = _solve_subproblem!(;
                model,
                columns,
                cuts,
                VV,
                instance,
                tol,
                silent,
                it,
                i,
                λ,
                μ,
                root_delays,
                delay_cost_function,
                backward,
                partial_bound,
                current_bound,
                λ_val,
                μ_val,
                FF,
                BF,
                order,
            )
            cp_sum += cp
            if cut_was_added
                done = false
                add_all_columns || break  # break if we add only one column per iteration
            end
        end
        lb = max(lb, cp_sum + sum(λ_val))
        push!(lb_history, lb)
        push!(ub_history, ub)
    end

    optimize!(model)
    obj = objective_value(model)
    dual_values = dual.(cuts)

    y_val_relax = retrieve_relaxation_solution_from_columns_and_duals(
        instance, columns, dual_values
    )

    return (;
        columns,
        obj,
        model,
        VV,
        y_val_relax,
        dual_values,
        ub_history,
        lb_history,
        feasible=true,
    )
end

"""
$TYPEDSIGNATURES

Take a vector of `Route`s and return the solution of the restricted master problem
heuristic for the stochastic tail assignment problem.
"""
function stochastic_column_heuristic(
    instance::ActivitySchedule,
    columns::Vector{Route},
    root_delays::AbstractMatrix;
    model_builder=highs_model,
    silent=true,
    delay_cost_function,
    bin=true,
    time_limit=nothing,
    warm_start=false,
)
    function column_cost(p)
        return full_cost(p, root_delays, instance; delay_cost_function)
    end

    return column_heuristic(
        instance, columns; model_builder, silent, column_cost, bin, time_limit, warm_start
    )
end

"""
$TYPEDSIGNATURES

Compute compact primal relaxation solution from columns and duals.
"""
function retrieve_relaxation_solution_from_columns_and_duals(
    instance::AbstractSchedule, columns, dual_values
)
    arc_index, _ = compute_arc_index(instance)
    y_val = zeros(length(values(arc_index)))
    for (route, dual_val) in zip(columns, dual_values)
        u = route[1]
        i = route.immat_index
        for v in route[2:length(route)]
            y_val[arc_index[u, v, i]] += dual_val
            u = v
        end
    end
    return y_val
end
