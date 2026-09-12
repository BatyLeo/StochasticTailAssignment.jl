"""
$TYPEDEF

Route data structure with useful maintenance information.

# Fields
$TYPEDFIELDS
"""
struct RouteWithMaintenance <: AbstractRoute
    "a route containing only leg indices"
    route::Route
    "for each leg in `route`, ordered list of maintenance indices before the next leg"
    maintenances::Vector{Vector{Int}}
end

immat_index(route::RouteWithMaintenance) = immat_index(route.route)

Base.lastindex(route::RouteWithMaintenance) = lastindex(route.route)

function Base.getindex(route::RouteWithMaintenance, idx...)
    return getindex(route.route, idx...)
end

function Base.getindex(A::AbstractArray, route::RouteWithMaintenance)
    return getindex(A, route.route)
end

function Base.setindex!(A::AbstractArray, v, route::RouteWithMaintenance)
    return setindex!(A, v, route.route)
end

function Base.view(route::RouteWithMaintenance, idx...)
    return view(route.route, idx...)
end

function RouteWithMaintenance(route::Route, schedule::ActivitySchedule)
    cleaned_route = Int[]
    maintenances = Vector{Int}[]
    L = nb_legs(schedule)
    for r in route.route
        if !is_maintenance(schedule, r)
            push!(cleaned_route, r)
            push!(maintenances, Int[])
        elseif is_maintenance(schedule, r) && length(cleaned_route) > 0
            push!(maintenances[end], r - L)
        end
    end
    return RouteWithMaintenance(Route(route, cleaned_route), maintenances)
end

@inline function RouteWithMaintenance(route::RouteWithMaintenance, ::ActivitySchedule)
    return route
end

@inline function RouteWithMaintenance(
    routes::Vector{RouteWithMaintenance}, ::ActivitySchedule
)
    return routes
end

function RouteWithMaintenance(routes::Vector{Vector{Int}}, schedule::ActivitySchedule)
    @assert length(routes) == nb_immats(schedule)
    return [RouteWithMaintenance(route, schedule) for route in routes_from_routes(routes)]
end

function RouteWithMaintenance(routes::Vector{Route}, schedule::ActivitySchedule)
    return [RouteWithMaintenance(route, schedule) for route in routes]
end

Base.iterate(x::RouteWithMaintenance) = iterate(x.route)
Base.iterate(x::RouteWithMaintenance, state) = iterate(x.route, state)
Base.length(x::RouteWithMaintenance) = length(x.route)
