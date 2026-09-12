struct Route <: AbstractRoute
    immat_index::Int
    route::Vector{Int}
end

immat_index(route::Route) = route.immat_index

function Base.getindex(route::Route, idx...)
    return getindex(route.route, idx...)
end

function Base.getindex(A::AbstractArray, route::Route)
    return getindex(A, route.route)
end

function Base.setindex!(A::AbstractArray, v, route::Route)
    return setindex!(A, v, route.route)
end

Base.lastindex(route::Route) = lastindex(route.route)

function Base.view(route::Route, idx...)
    return view(route.route, idx...)
end

function Route(route::Route, new_route::Vector{Int})
    return Route(route.immat_index, new_route)
end

function routes_from_routes(routes::Vector{Vector{Int}})
    return [Route(i, route) for (i, route) in enumerate(routes)]
end

Base.iterate(x::Route) = iterate(x.route)
Base.iterate(x::Route, state) = iterate(x.route, state)
Base.length(x::Route) = length(x.route)

"""
$TYPEDSIGNATURES

Returns an ordered vector of leg activity indices in `route` (i.e. filtering out maintenances).
"""
function leg_indices_from_route(route, instance::AbstractSchedule)
    # L = nb_legs(instance)
    L = leg_indices(instance)
    return [l for l in route if l in L]
end

function Base.:(==)(x::Route, y::Route)
    return x.immat_index == y.immat_index &&
           length(x) == length(y) &&
           all(x.route .== y.route)
end
