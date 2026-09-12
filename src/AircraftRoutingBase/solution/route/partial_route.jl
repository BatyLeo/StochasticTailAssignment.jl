"""
$TYPEDEF

This data structure enables defining a partial route with a time window.
This is used to split a route between maintenances, to only consider legs.

# Fields
$TYPEDFIELDS
"""
struct PartialRoute
    "list of leg indices corresponding to this partial route"
    leg_route::Route
    "start of the time window for this partial route"
    start_time::Dates.DateTime
    "end of the time window for this partial route"
    end_time::Dates.DateTime
end

Base.iterate(x::PartialRoute) = iterate(x.leg_route)
Base.iterate(x::PartialRoute, state) = iterate(x.leg_route, state)
Base.length(x::PartialRoute) = length(x.leg_route)

function Base.getindex(route::PartialRoute, idx...)
    return getindex(route.leg_route, idx...)
end

function Base.getindex(A::AbstractArray, route::PartialRoute)
    return getindex(A, route.leg_route)
end

function Base.setindex!(A::AbstractArray, v, route::PartialRoute)
    return setindex!(A, v, route.leg_route)
end

function Base.view(route::PartialRoute, idx...)
    return view(route.leg_route, idx...)
end

"""
$TYPEDSIGNATURES

Convert a given feasible route into a vector of partial routes splitting the maintenances.
"""
function route_to_partial_routes(route::Route, schedule::ActivitySchedule)
    cleaned_route = PartialRoute[]
    immat = route.immat_index
    L = nb_legs(schedule)
    current_index = 1 # current index in the route
    first_index = 1   # start of the route
    a, b = get_time_horizon(schedule)
    start_t = a
    while current_index <= length(route)
        r = route[current_index]
        m_activity = get_activity(schedule, r)
        if is_maintenance(schedule, r)
            if current_index - first_index > 0
                end_t = start_time(m_activity)
                push!(
                    cleaned_route,
                    PartialRoute(
                        Route(immat, route[first_index:(current_index - 1)]), start_t, end_t
                    ),
                )
            end
            first_index = current_index + 1
            start_t = end_time(m_activity)
        end
        current_index += 1
    end
    if current_index - first_index > 0
        end_t = b
        push!(
            cleaned_route,
            PartialRoute(
                Route(immat, route[first_index:(current_index - 1)]), start_t, end_t
            ),
        )
    end

    return cleaned_route
end

"""
$TYPEDSIGNATURES

Check if two partial routes intersect in time, i.e. if they are elegible for a swap.
"""
function is_intersection(p1::PartialRoute, p2::PartialRoute)
    return !(p1.end_time < p2.start_time || p2.end_time < p1.start_time)
end
