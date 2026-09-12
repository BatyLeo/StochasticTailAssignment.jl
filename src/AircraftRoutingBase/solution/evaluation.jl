"""
$TYPEDSIGNATURES

Compute the total cost of operating all activities in the routes `routes`.
"""
function operational_cost(
    routes::Vector{Vector{Int}},
    instance::ActivitySchedule;
    include_chaining_costs=instance.include_chaining_costs,
)
    @assert length(routes) == nb_immats(instance)
    res = 0.0
    for (i, route) in enumerate(routes)
        for (j, v) in enumerate(route)
            res += activity_cost(instance, v, i)
            if include_chaining_costs && j > 1
                res += chaining_cost(instance, route[j - 1], v)
            end
        end
    end
    return res
end

function operational_cost(
    route::Route,
    instance::ActivitySchedule;
    include_chaining_costs=instance.include_chaining_costs,
)
    if length(route) == 0
        return 0.0
    end
    activity_costs = sum(activity_cost(instance, v, immat_index(route)) for v in route)
    chaining_costs = if include_chaining_costs && length(route) > 1
        sum(chaining_cost(instance, u, v) for (u, v) in partition(route, 2, 1))
    else
        0.0
    end
    return chaining_costs + activity_costs
end

function operational_cost(
    routes::Vector{Route},
    instance::ActivitySchedule;
    include_chaining_costs=instance.include_chaining_costs,
)
    return sum(
        operational_cost(route, instance; include_chaining_costs) for route in routes
    )
end

function operational_cost(route_s, instance::MaskedSchedule; kwargs...)
    return operational_cost(route_s, instance.schedule; kwargs...)
end

"""
$TYPEDSIGNATURES

Check if a given route is feasible.
"""
function is_feasible(route::Route, schedule::ActivitySchedule; verbose=true)
    (; TTM_factor) = schedule
    immat_index = route.immat_index
    immat = schedule.immats[immat_index]
    L = nb_legs(schedule)
    maintenance_indices = get_maintenance_indices(schedule, immat_index)

    # Check if all maintenances are operated
    for mi in maintenance_indices
        if !(mi in route)
            verbose && @warn "Missing maintenance $(mi) in $(route)"
            return false
        end
    end

    i = 1
    previous_activity = get_activity(schedule, route[i])

    # check if maintenance is not for another immat
    if is_maintenance(previous_activity)
        if previous_activity.immat != immat.id
            verbose && @warn "Wrong maintenance $(immat.id) / $(previous_activity.immat)"
            return false
        end
    end

    while i < length(route)
        i += 1
        next_activity = get_activity(schedule, route[i])
        if is_maintenance(next_activity)
            if next_activity.immat != immat.id
                verbose &&
                    @warn "Wrong maintenance $(immat.id) / $(next_activity.immat) $(route[i])"
                return false
            end
        end
        # Check if chaining is valid
        if !is_valid_chaining(previous_activity, next_activity; TTM_factor)
            verbose && @warn "Invalid chaining"
            return false
        end
        previous_activity = next_activity
    end

    last_activity = immat.last_activity_id
    if haskey(schedule.graph, last_activity) && last_activity != "s"
        if code_for(schedule, last_activity) != route[1]
            verbose && @warn "Wrong last activity $(last_activity) / $(route[1])"
            return false
        end
    end
    return true
end

function is_feasible(route::Vector{Int}, immat_index::Int, schedule::ActivitySchedule)
    return is_feasible(Route(immat_index, route), schedule)
end

"""
$TYPEDSIGNATURES

Check if routes are feasible.
"""
function is_feasible(routes::Vector{Route}, schedule::ActivitySchedule; verbose=true)
    is_here = falses(nb_activities(schedule))

    for route in routes
        if !is_feasible(route, schedule; verbose)
            return false
        end
        for r in route
            is_here[r] = true
        end
    end
    res = all(is_here)
    verbose &&
        !res &&
        @warn "Some activities are missing: $([i for i in 1:nb_activities(schedule) if !is_here[i]])"
    return res
end

function is_feasible(routes::Vector{Vector{Int}}, schedule::ActivitySchedule; verbose=true)
    return is_feasible(routes_from_routes(routes), schedule; verbose)
end

function is_feasible(route_s, schedule::MaskedSchedule; verbose=true)
    return is_feasible(route_s, schedule.schedule; verbose)
end
