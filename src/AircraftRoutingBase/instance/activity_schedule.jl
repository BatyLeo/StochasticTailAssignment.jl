"""
$TYPEDEF

Data structure defining a schedule of activities.
Two implementations :
- `ActivitySchedule` : the main data structure, containing all the information about the schedule.
- `MaskedSchedule` : a masked version of the `ActivitySchedule`, where some activities are removed.
"""
abstract type AbstractSchedule end

"""
$TYPEDEF

Data structure defining a schedule of activities.
It corresponds to an instance of the aircraft routing (fleet assignment) problem.

# Fields
$TYPEDFIELDS
"""
struct ActivitySchedule{
    G<:MetaGraphsNext.MetaGraph,G1<:MetaGraphsNext.MetaGraph,L<:AbstractLeg,A
} <: AbstractSchedule
    "all available immatriculations"
    immats::Vector{Immat}
    "all legs that need to be operated in the schedule"
    legs::Vector{L}
    "all maintenance task that need to be performed in the schedule"
    maintenances::Vector{Maintenance}
    "all forced chaining enforced in the schedule"
    forced_chainings::Vector{ForcedChaining}
    "dependency (acyclic) digraph between activities"
    graph::G
    "specialized digraph for each immatriculation, removing useless edges"
    immat_graphs::Vector{G1}
    "minimum turn time multiplier that was used to compute graph edges"
    TTM_factor::Float64
    "whether to include chaining costs in the operational costs"
    include_chaining_costs::Bool
    "[u, v, i] = arc index for learning"
    arc_index::A
    "number of (u, v, i) not connected to s or t"
    nb_interior_arcs::Int
end

function build_graph(
    legs::Vector{<:AbstractLeg},
    maintenances::Vector{Maintenance},
    forced_chainings::Vector{ForcedChaining},
    immat_leg_cost;
    empty_graph=false,
    TTM_factor,
    tractage_cost=1000.0,
    sparsification_threshold=typemax(Int),
)
    activities = (legs, maintenances)
    I = size(immat_leg_cost, 1)

    metagraph = MetaGraphsNext.MetaGraph(
        Graphs.DiGraph();
        label_type=String,  # activity id
        vertex_data_type=Vector{Float64},  # vector of costs for each immat
        edge_data_type=Float64,  # connection cost
        graph_data="Routing graph",
    )

    if empty_graph
        return metagraph
    end

    # Create a vertex for each activity, starting by legs
    for activity_set in activities
        for activity in activity_set
            metagraph[id(activity)] = fill(0.0, I)
        end
    end
    for (index_1, activity_1) in enumerate(legs)
        id1 = id(activity_1)
        metagraph[id1] .= immat_leg_cost[:, index_1]
    end

    # Create vertices for both dummy vertices
    metagraph["s"] = fill(0.0, I)
    metagraph["t"] = fill(0.0, I)
    metagraph["s", "t"] = 0.0

    forced_chaining_inn_ids = [f.first_activity_id for f in forced_chainings]
    forced_chaining_out_ids = [f.second_activity_id for f in forced_chainings]
    fc = Dict(f.first_activity_id => f.second_activity_id for f in forced_chainings)

    for activity_set in activities
        for activity_1 in activity_set
            id1 = id(activity_1)

            if !(id1 in forced_chaining_inn_ids)
                metagraph[id1, "t"] = 0.0
            end
            if !(id1 in forced_chaining_out_ids)
                metagraph["s", id1] = 0.0
            end

            for activity_set_2 in activities
                for activity_2 in activity_set_2
                    id2 = id(activity_2)
                    if id1 == id2
                        continue
                    end
                    if id1 in forced_chaining_inn_ids
                        if id2 == fc[id1]
                            @assert is_valid_chaining(activity_1, activity_2; TTM_factor)
                            metagraph[id1, id2] = chaining_cost(
                                activity_1, activity_2; tractage_cost
                            )
                            break
                        else
                            continue
                        end
                    end
                    # else
                    # start_cdg_ory =
                    #     start_airport(activity_2) in ["CDG", "ORY"] &&
                    #     !is_maintenance(activity_1)
                    # if is_valid_chaining(
                    #     activity_1, activity_2; TTM_factor=TTM_factor * start_cdg_ory
                    # )
                    #     metagraph[id1, id2] = chaining_cost(activity_1, activity_2; tractage_cost)
                    # end
                    if is_valid_chaining(activity_1, activity_2; TTM_factor)
                        if slack(activity_1, activity_2) >= sparsification_threshold # remove the graph if more than threshold slack
                            continue
                        end
                        metagraph[id1, id2] = chaining_cost(
                            activity_1, activity_2; tractage_cost
                        )
                    end
                end
            end
        end
    end
    return metagraph
end

"""
$TYPEDSIGNATURES

Build a subgraph for each immatriculation, corresponding to the routing graph for that immatriculation.
Use the same base graph for all immat graphs, in order to have the same vertex indices.
"""
function build_immat_graphs(
    graph::MetaGraphsNext.MetaGraph,
    immats::Vector{Immat},
    maintenances::Vector{Maintenance},
    legs::Vector{<:AbstractLeg};
    empty_graph=false,
)
    if empty_graph
        return [graph for _ in immats]
    end
    I = length(immats)
    L = length(legs)

    maintenances_by_immat = map(1:I) do i
        return get_maintenance_indices(maintenances, immats, i) .+ L
    end

    metagraphs = map(1:I) do i
        return MetaGraphsNext.MetaGraph(
            Graphs.DiGraph();
            label_type=String,        # activity id
            vertex_data_type=Float64, # cost for given immat
            edge_data_type=Float64,   # connection cost
            graph_data="Routing graph for immat $i",
        )
    end

    # Copy all vertices, and the corresponding immat dependent costs
    for v in vertices(graph)
        v_id = label_for(graph, v)
        for i in 1:I
            metagraphs[i][v_id] = graph[v_id][i]
        end
    end

    s = code_for(graph, "s")
    t = code_for(graph, "t")

    is_useful = falses(I, L)
    for i in 1:I
        indices = maintenances_by_immat[i]
        for u in 1:L
            if length(indices) == 0
                is_useful[i, u] = true
            elseif has_path(graph, u, indices[1]) || has_path(graph, indices[end], u)
                is_useful[i, u] = true
            else
                MM = length(indices) - 1
                for j in 1:MM
                    if has_path(graph, indices[j], u) && has_path(graph, u, indices[j + 1])
                        is_useful[i, u] = true
                        break
                    end
                end
            end
        end
    end

    for arc in edges(graph)
        u, v = src(arc), dst(arc)
        for i in 1:I
            u_id, v_id = label_for(graph, u), label_for(graph, v)
            indices = maintenances_by_immat[i]

            # prune with last activity info: only keep (s, v) where v is the last activity (if there is one)
            last_activity_id = immats[i].last_activity_id
            if u == s &&
                last_activity_id != "s" &&
                haskey(graph, last_activity_id) &&
                v_id != last_activity_id
                continue
            end

            # skip edge if encountering a maintenance from another immat
            if u > L && u != s
                u_activity = maintenances[u - L]
                if u_activity.immat != immats[i].id
                    continue
                end
            end
            if v > L && v != t
                v_activity = maintenances[v - L]
                if v_activity.immat != immats[i].id
                    continue
                end
            end

            # when both are either useful, maintenances or dummy vertices, check if we do not skip any maintenance
            if (u > L || is_useful[i, u]) && (v > L || is_useful[i, v])
                is_useful_arc = false
                if length(indices) == 0
                    is_useful_arc = true
                elseif v == indices[1] ||
                    u == indices[end] ||
                    has_path(graph, v, indices[1]) ||
                    has_path(graph, indices[end], u)
                    is_useful_arc = true
                else
                    MM = length(indices) - 1
                    for j in 1:MM
                        if (u == indices[j] || has_path(graph, indices[j], u)) &&
                            (v == indices[j + 1] || has_path(graph, v, indices[j + 1]))
                            is_useful_arc = true
                            break
                        end
                    end
                end

                if is_useful_arc
                    metagraphs[i][u_id, v_id] = graph[u_id, v_id]
                end
            end
        end
    end
    return metagraphs
end

"""
$TYPEDSIGNATURES

Constructor for an [`ActivitySchedule`](@ref).
"""
function ActivitySchedule(;
    immats::Vector{Immat},
    legs::Vector{<:AbstractLeg},
    maintenances::Vector{Maintenance}=Maintenance[],
    forced_chainings::Vector{ForcedChaining}=ForcedChaining[],
    fuel_cost_per_kg=0.7,
    standard_consumption::Dict{String,Float64},
    tractage_cost::Float64=1000.0,
    empty_graph=false,
    TTM_factor=0.0,
    include_chaining_costs=true,
    store_arc_index=false,
    sparsification_threshold=typemax(Int),
)
    immat_leg_cost = map(Iterators.product(immats, legs)) do (immat, leg)
        leg_duration = duration(leg)
        t = aircraft_type(leg)
        consumption = if haskey(standard_consumption, t)
            standard_consumption[t]
        else
            standard_consumption[""]
        end
        return consumption *
               (0.0 + immat.fuel_factor / 100) *
               leg_duration *
               fuel_cost_per_kg
    end
    graph = build_graph(
        legs,
        maintenances,
        forced_chainings,
        immat_leg_cost;
        empty_graph,
        TTM_factor,
        tractage_cost,
        sparsification_threshold,
    )
    graphs = build_immat_graphs(graph, immats, maintenances, legs; empty_graph)

    arc_index, nb_interior_arcs = if store_arc_index
        compute_arc_index(graphs, graph)
    else
        nothing, 0
    end

    return ActivitySchedule(
        immats,
        legs,
        maintenances,
        forced_chainings,
        graph,
        graphs,
        TTM_factor,
        include_chaining_costs,
        arc_index,
        nb_interior_arcs,
    )
end

"""
$TYPEDSIGNATURES

Retrieve the number of immatriculations in the schedule.
"""
nb_immats(instance::ActivitySchedule) = length(instance.immats)

"""
$TYPEDSIGNATURES

Retrieve the number of legs in the schedule.
"""
nb_legs(instance::ActivitySchedule) = length(instance.legs)

"""
$TYPEDSIGNATURES

Retrieve the number of maintenances in the schedule.
"""
nb_maintenances(instance::ActivitySchedule) = length(instance.maintenances)

"""
$TYPEDSIGNATURES

Retrieve the total number of activities (legs + maintenances) in the schedule.
"""
nb_activities(instance::ActivitySchedule) = nb_legs(instance) + nb_maintenances(instance)

"""
$TYPEDSIGNATURES

Retrieve all activity indices in the schedule. Useful for iteration.
"""
activity_indices(instance::ActivitySchedule) = 1:nb_activities(instance)

"""
$TYPEDSIGNATURES

Retrieve all leg indices in the schedule. Useful for iteration.
"""
leg_indices(instance::ActivitySchedule) = 1:nb_legs(instance)

"""
$TYPEDSIGNATURES

Retrieve all immatriculation indices in the schedule. Useful for iteration.
"""
immat_indices(instance::ActivitySchedule) = 1:nb_immats(instance)

does_include_chaining_costs(instance::ActivitySchedule) = instance.include_chaining_costs

# (Meta)Graphs methods extension
Graphs.nv(instance::AbstractSchedule) = Graphs.nv(instance.graph)
Graphs.ne(instance::AbstractSchedule) = Graphs.ne(instance.graph)
Graphs.edges(instance::AbstractSchedule) = Graphs.edges(instance.graph)

function Graphs.outneighbors(instance::ActivitySchedule, index::Int)
    return Graphs.outneighbors(instance.graph, index)
end
function Graphs.inneighbors(instance::ActivitySchedule, index::Int)
    return Graphs.inneighbors(instance.graph, index)
end
function Graphs.outneighbors(instance::ActivitySchedule, id::String)
    return Graphs.outneighbors(instance.graph, code_for(instance, id))
end
function Graphs.inneighbors(instance::ActivitySchedule, id::String)
    return Graphs.inneighbors(instance.graph, code_for(instance, id))
end
function MetaGraphsNext.code_for(instance::ActivitySchedule, id::String)
    return MetaGraphsNext.code_for(instance.graph, id)
end
function MetaGraphsNext.label_for(instance::ActivitySchedule, index::Int)
    return MetaGraphsNext.label_for(instance.graph, index)
end

"""
$TYPEDSIGNATURES

Retrieve the vertex code of the source vertex `s` in the graph.
"""
function get_s(instance::AbstractSchedule)
    return code_for(instance.graph, "s")
end

"""
$TYPEDSIGNATURES

Retrieve the vertex code of the sink vertex `t` in the graph.
"""
function get_t(instance::AbstractSchedule)
    return code_for(instance.graph, "t")
end

"""
$TYPEDSIGNATURES

Retrieve the vertex code of the source vertex `s` in the graph.
"""
function get_s(graph::MetaGraphsNext.MetaGraph)
    return code_for(graph, "s")
end

"""
$TYPEDSIGNATURES

Retrieve the vertex code of the sink vertex `t` in the graph.
"""
function get_t(graph::MetaGraphsNext.MetaGraph)
    return code_for(graph, "t")
end

"""
$TYPEDSIGNATURES

Retrieve the cost of operating activity (usually a leg) corresponding to index `v` with
aircraft immatriculation (index) `i` in the graph.
"""
function activity_cost(instance::AbstractSchedule, v, i)
    (; graph) = instance
    return graph[label_for(graph, v)][i]
end

"""
$TYPEDSIGNATURES

Retrieve (from the graph) the cost of chaining from vertex `u` to vertex `v`.
"""
function chaining_cost(instance::AbstractSchedule, u, v)
    (; graph) = instance
    return graph[label_for(graph, u), label_for(graph, v)]
end

"""
$TYPEDSIGNATURES

Compute the cost of an edge as the sum of the operational cost of the destination plus the eventual chaining cost of the edge.
"""
function edge_cost(
    instance::AbstractSchedule,
    u::Int,
    v::Int,
    i::Int;
    include_chaining_costs=does_include_chaining_costs(instance),
)
    return activity_cost(instance, v, i) +
           include_chaining_costs * chaining_cost(instance, u, v)
end

"""
$TYPEDSIGNATURES

Compute the cost of an edge as the sum of the operational cost of the origin plus the chaining cost of the edge.
"""
function edge_cost_origin(
    instance::AbstractSchedule,
    u::Int,
    v::Int,
    i::Int;
    include_chaining_costs=does_include_chaining_costs(instance),
)
    return activity_cost(instance, u, i) +
           include_chaining_costs * chaining_cost(instance, u, v)
end

"""
$TYPEDSIGNATURES

Check if given activity index `activity_index` corresponds to a leg activity.
"""
function is_leg(schedule::AbstractSchedule, activity_index::Int)
    return activity_index in leg_indices(schedule)
end

"""
$TYPEDSIGNATURES

Check if given activity index `activity_index` corresponds to a maintenance.
"""
function is_maintenance(schedule::AbstractSchedule, activity_index::Int; safe_mode=true)
    safe_mode && @assert !is_s(schedule, activity_index) && !is_t(schedule, activity_index)
    return !(activity_index in leg_indices(schedule))
end

"""
$TYPEDSIGNATURES

Check if `activity_index` is the source of the graph.
"""
function is_s(schedule, activity_index::Int)
    return activity_index == get_s(schedule)
end

"""
$TYPEDSIGNATURES

Check if `activity_index` is the sink of the graph.
"""
function is_t(schedule, activity_index::Int)
    return activity_index == get_t(schedule)
end

"""
$TYPEDSIGNATURES

Check if `activity_index` is either one of the two dummy vertices of the graph.
"""
function is_dummy_vertex(schedule::ActivitySchedule, activity_index::Int)
    return is_s(schedule, activity_index) || is_t(schedule, activity_index)
end

"""
$TYPEDSIGNATURES

Retrieve the activity corresponding to the given index.

!!! warning
    This method might be type instable. It can return either a `Leg` or a `Maintenance`, depending on `activity_index` value.
"""
function get_activity(schedule::ActivitySchedule, activity_index::Int)
    (; maintenances, legs) = schedule
    L = nb_legs(schedule)
    return if is_maintenance(schedule, activity_index)
        maintenances[activity_index - L]
    else
        legs[activity_index]
    end
end

"""
$TYPEDSIGNATURES

Return last activity index of immat if it exists, else index of s.
"""
function get_last_activity_index(schedule::ActivitySchedule, immat_index::Int)
    (; graph, immats) = schedule
    last_activity_id = immats[immat_index].last_activity_id
    if haskey(graph, last_activity_id) && last_activity_id != "s"
        return code_for(graph, last_activity_id)
    else
        return get_s(schedule)
    end
end

function get_maintenance_indices(
    maintenances::Vector{Maintenance},
    immats::Vector{Immat},
    immat_index::Int;
    sorted::Bool=true,
)
    M = length(maintenances)
    maintenance_indices = [
        m for m in 1:M if maintenances[m].immat == immats[immat_index].id
    ]
    if sorted
        immat_maintenances = maintenances[maintenance_indices]
        return maintenance_indices[sortperm(immat_maintenances; lt=is_time_compatible)]
    end
    # else
    return maintenance_indices
end

"""
$TYPEDSIGNATURES

Return a vector of node indices corresponding to all maintenances affected to aircraft `immat_index`.
Sorted by start time by default.
"""
function get_maintenance_indices(schedule::ActivitySchedule, immat_index::Int; sorted=true)
    (; maintenances, immats) = schedule
    L = nb_legs(schedule)
    return get_maintenance_indices(maintenances, immats, immat_index; sorted) .+ L
end

"""
$TYPEDSIGNATURES

Returns a boolean mask of the edges that connect two activity (i.e. neither of the dummy vertices).
"""
function compute_edge_mask(schedule::ActivitySchedule)
    return [(!is_s(schedule, src(e)) && !is_t(schedule, dst(e))) for e in edges(schedule)]
end

"""
$TYPEDSIGNATURES

Returns the aircraft type of the first leg in the schedule, assuming all legs have the same aircraft type.
"""
function aircraft_type(schedule::ActivitySchedule)
    return aircraft_type(schedule.legs[1])
end

function get_time_horizon(schedule::ActivitySchedule)
    N = nb_activities(schedule)
    return minimum(start_time(get_activity(schedule, a)) for a in 1:N),
    maximum([end_time(get_activity(schedule, a)) for a in 1:N])
end

"""
$TYPEDSIGNATURES

Compute the arc index for the given graph and schedule.
[u, v, i] -> arc index
"""
function compute_arc_index(immat_graphs, graph)
    arc_index = Dict{Tuple{Int,Int,Int},Int}()
    c = 0
    for i in eachindex(immat_graphs)
        for arc in edges(immat_graphs[i])
            if !is_s(graph, src(arc)) && !is_t(graph, dst(arc))
                c += 1
                arc_index[src(arc), dst(arc), i] = c
            end
        end
    end
    d = 0
    for i in eachindex(immat_graphs)
        for arc in edges(immat_graphs[i])
            if is_s(graph, src(arc)) || is_t(graph, dst(arc))
                d += 1
                arc_index[src(arc), dst(arc), i] = c + d
            end
        end
    end
    return arc_index, c
end

"""
$TYPEDSIGNATURES

Compute the arc index for the given schedule.
"""
function compute_arc_index(schedule::AbstractSchedule)
    return compute_arc_index(schedule.immat_graphs, schedule.graph)
end
