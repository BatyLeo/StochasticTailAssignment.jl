"""
$TYPEDEF

Wrapper over [`ActivitySchedule`](@ref) to represent a schedule where some routes are fixed.
For each route, the associared immat does not need to be considered anymore, as well as the activities in the route.

# Fields
$TYPEDFIELDS
"""
struct MaskedSchedule{
    T<:ActivitySchedule,G<:MetaGraphsNext.MetaGraph,G1<:MetaGraphsNext.MetaGraph
} <: AbstractSchedule
    "associated schedule"
    schedule::T
    "mask of immats that are not fixed"
    immat_mask::BitVector
    "mask of activities that are not fixed"
    activity_mask::BitVector
    "mask of legs that are not fixed"
    leg_mask
    "connection graph"
    graph::G
    "connection sub graphs"
    immat_graphs::Vector{G1}
end

"""
$TYPEDSIGNATURES

Remove edges connected to masked activities (this allows keeping consistent vertex indexing)
"""
function mask_graph!(graph, route::Route; verbose=false)
    for edge in collect(edges(graph))
        if src(edge) in route || dst(edge) in route
            verbose && @info "Removing edge $(src(edge)) -> $(dst(edge))"
            @assert rem_edge!(graph, src(edge), dst(edge))
        end
    end
end

function MaskedSchedule(schedule::ActivitySchedule)
    return MaskedSchedule(
        schedule,
        trues(nb_immats(schedule)),
        trues(nb_activities(schedule)),
        trues(nb_legs(schedule)),
        deepcopy(schedule.graph),
        deepcopy(schedule.immat_graphs),
    )
end

function MaskedSchedule(schedule::ActivitySchedule, route::Route)
    immat_mask = trues(nb_immats(schedule))
    immat_mask[route.immat_index] = false

    activity_mask = trues(nb_activities(schedule))
    leg_mask = trues(nb_legs(schedule))
    for activity_index in route
        activity_mask[activity_index] = false
        if is_leg(schedule, activity_index)
            leg_mask[activity_index] = false
        end
    end

    graph = deepcopy(schedule.graph)
    immat_graphs = deepcopy(schedule.immat_graphs)

    # Remove edges connected to masked activities (this allows keeping consistent vertex indexing)
    mask_graph!(graph, route)
    for g in immat_graphs
        mask_graph!(g, route)
    end

    return MaskedSchedule(
        schedule, immat_mask, activity_mask, leg_mask, graph, immat_graphs
    )
end

function mask_schedule!(schedule::MaskedSchedule, route::Route)
    schedule.immat_mask[route.immat_index] = false
    for activity_index in route
        schedule.activity_mask[activity_index] = false
        if is_leg(schedule.schedule, activity_index)
            schedule.leg_mask[activity_index] = false
        end
    end

    mask_graph!(schedule.graph, route)
    for g in schedule.immat_graphs
        mask_graph!(g, route)
    end

    return nothing
end

function mask_schedule(schedule::MaskedSchedule, route::Route)
    new_schedule = deepcopy(schedule)
    mask_schedule!(new_schedule, route)
    return new_schedule
end

function MaskedSchedule(schedule::MaskedSchedule, route::Route)
    return mask_schedule(schedule, route)
end

function MaskedSchedule(schedule::ActivitySchedule, routes::AbstractVector{Route})
    masked_schedule = MaskedSchedule(schedule, routes[1])
    for route in routes[2:end]
        mask_schedule!(masked_schedule, route)
    end
    return masked_schedule
end

function immat_indices(schedule::MaskedSchedule)
    return (1:nb_immats(schedule.schedule))[schedule.immat_mask]
end
function activity_indices(schedule::MaskedSchedule)
    return (1:nb_activities(schedule.schedule))[schedule.activity_mask]
end
leg_indices(schedule::MaskedSchedule) = (1:nb_legs(schedule.schedule))[schedule.leg_mask]

nb_immats(schedule::MaskedSchedule) = sum(schedule.immat_mask)
nb_activities(schedule::MaskedSchedule) = sum(schedule.activity_mask)
nb_legs(schedule::MaskedSchedule) = sum(schedule.leg_mask)
nb_maintenances(schedule::MaskedSchedule) = nb_activities(schedule) - nb_legs(schedule)

function does_include_chaining_costs(instance::MaskedSchedule)
    return does_include_chaining_costs(instance.schedule)
end

function get_activity(schedule::MaskedSchedule, activity_index::Int)
    return get_activity(schedule.schedule, activity_index)
end
