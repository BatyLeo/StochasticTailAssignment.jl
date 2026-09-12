"""
$TYPEDSIGNATURES

Encode input routes to a MIP solution.
"""
function encode_routes_to_solution(routes, instance)
    I = nb_immats(instance)
    V = nv(instance.graph)
    y_val = zeros(Int, I, V, V)
    s = get_s(instance)
    t = get_t(instance)
    for i in 1:I
        R = length(routes[i])
        y_val[i, s, routes[i][1]] = 1
        y_val[i, routes[i][R], t] = 1
        for (u, v) in zip(routes[i], routes[i][2:R])
            y_val[i, u, v] = 1
        end
    end
    return y_val
end

"""
$TYPEDSIGNATURES

Build JuMP model for [`solve_stochastic_aircraft_routing_mip`](@ref).
"""
function build_stochastic_model(
    instance::ActivitySchedule,
    root_delays::AbstractMatrix,
    delay_cost_function::PiecewiseLinearFunction;
    model_builder=highs_model,
    include_chaining_costs=instance.include_chaining_costs,
    silent=false,
    time_limit=nothing,
    warm_start_routes=nothing,
)
    graph = instance.graph
    immat_graphs = instance.immat_graphs
    V = nv(graph)
    L = nb_legs(instance)
    I = nb_immats(instance)
    s = get_s(instance)
    t = get_t(instance)

    Ω = size(root_delays, 1)

    model = model_builder()
    silent && set_silent(model)

    if !isnothing(time_limit)
        set_time_limit_sec(model, time_limit)
    end

    # decision variables
    model[:y] = if !isnothing(warm_start_routes)
        y_val = encode_routes_to_solution(warm_start_routes, instance)
        @variable(
            model,
            y[i in 1:I, u in 1:V, v in outneighbors(immat_graphs[i], u)],
            Bin,
            start = y_val[i, u, v]
        )
    else
        @variable(model, y[i in 1:I, u in 1:V, v in outneighbors(immat_graphs[i], u)], Bin)
    end

    breakpoints = delay_cost_function.x
    adjusted_slopes = compute_slopes(delay_cost_function)
    J = length(adjusted_slopes)

    # maintenances do not have any root delays
    adjusted_root_delays = hcat(
        root_delays, fill(zero(eltype(root_delays)), Ω, nb_maintenances(instance))
    )

    # delay variables
    model[:d] = @variable(model, d[v in 1:V, ω in 1:Ω; v != s && v != t])

    # piecewise linear discretization variables
    model[:dj] = @variable(model, dj[v in 1:V, j in 1:J, ω in 1:Ω; v != s && v != t])

    @expression(
        model,
        z⁺[u in 1:V, i in 1:I],
        sum(y[i, u, v] for v in outneighbors(immat_graphs[i], u))
    )

    @expression(
        model,
        z⁻[v in 1:V, i in 1:I],
        sum(y[i, u, v] for u in inneighbors(immat_graphs[i], v))
    )

    @expression(
        model,
        operational_costs,
        sum(activity_cost(instance, v, i) * z⁺[v, i] for v in 1:V for i in 1:I)
    )

    @expression(
        model,
        chaining_costs,
        sum(
            chaining_cost(instance, u, v) * y[i, u, v] for i in 1:I for u in 1:V for
            v in outneighbors(immat_graphs[i], u)
        )
    )

    @expression(
        model,
        delay_costs,
        sum(
            adjusted_slopes[j] * dj[v, j, ω] for v in 1:L for j in 1:J for
            ω in 1:Ω if v != s && v != t
        ) / Ω
    )

    @objective(
        model,
        Min,
        delay_costs + operational_costs + include_chaining_costs * chaining_costs
    )

    # flow constraints
    @constraint(model, [i in 1:I], z⁺[s, i] == 1)
    @constraint(model, [i in 1:I], z⁻[t, i] == 1)
    @constraint(model, [v in 1:V, i in 1:I; v != s && v != t], z⁻[v, i] == z⁺[v, i])

    # set covering constraint: all activity must be performed
    @constraint(model, [v in 1:V; v != s && v != t], sum(z⁻[v, i] for i in 1:I) == 1)

    # Delay propagation constraints
    @constraint(
        model, [v in 1:V, ω in 1:Ω; v != s && v != t], d[v, ω] >= adjusted_root_delays[ω, v]
    )
    @constraint(
        model,
        [v in 1:V, ω in 1:Ω; v != s && v != t],
        d[v, ω] >=
            adjusted_root_delays[ω, v] + sum(
            y[i, u, v] * (
                d[u, ω] -
                slack_with_turn_time(get_activity(instance, u), get_activity(instance, v))
            ) for i in 1:I for u in inneighbors(immat_graphs[i], v) if u != s && u != t
        )
    )

    # Piecewise linear discretization constraints
    @constraint(
        model,
        [v in 1:V, ω in 1:Ω; v != s && v != t],
        d[v, ω] == sum(dj[v, j, ω] for j in 1:J)
    )
    @constraint(
        model,
        [v in 1:V, j in 2:(J - 1), ω in 1:Ω; v != s && v != t],
        dj[v, j, ω] <= (breakpoints[j] - breakpoints[j - 1])
    )
    @constraint(model, [v in 1:V, j in 2:J, ω in 1:Ω; v != s && v != t], dj[v, j, ω] >= 0)
    @constraint(model, [v in 1:V, ω in 1:Ω; v != s && v != t], dj[v, 1, ω] <= 0.0)

    return model
end

"""
$TYPEDSIGNATURES

Solve the stochastic tail assignment problem.
"""
function solve_stochastic_aircraft_routing_mip(
    instance::ActivitySchedule,
    root_delays,
    delay_cost_function::PiecewiseLinearFunction;
    silent=true,
    kwargs...,
)
    model = build_stochastic_model(
        instance, root_delays, delay_cost_function; silent, kwargs...
    )
    optimize!(model)
    status = termination_status(model)
    silent || @info status
    if has_values(model)
        y_val = value.(model[:y])
        return (;
            routes=decode_routes_from_solution(y_val, instance),
            obj=objective_value(model),
            status,
            time=solve_time(model),
            gap=relative_gap(model),
        )
    else
        return (;
            routes=Vector{Int}[],
            obj=Inf,
            status,
            time=solve_time(model),
            gap=relative_gap(model),
        )
    end
end
