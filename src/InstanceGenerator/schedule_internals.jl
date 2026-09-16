function _assign_spoke_distances(rng, spokes::Vector{String})
    return Dict(sp => (i - 1 + rand(rng)) / length(spokes) for (i, sp) in enumerate(spokes))
end

function _allocate_legs(rng, nb_legs::Int, nb_aircraft::Int)
    weights = [0.3 + rand(rng) for _ in 1:nb_aircraft]
    allocation = [max(1, round(Int, nb_legs * w / sum(weights))) for w in weights]
    diff = nb_legs - sum(allocation)
    for i in 1:abs(diff)
        allocation[((i - 1) % nb_aircraft) + 1] += sign(diff)
    end
    return allocation
end

function _sample_turn_time(rng, min_turn::Int, max_turn::Int, tight_fraction::Float64)
    if rand(rng) < tight_fraction && min_turn > 1
        return rand(rng, max(1, min_turn ÷ 2):(min_turn - 1))
    end
    return rand(rng, min_turn:max_turn)
end

function _make_leg(index::Int, dep, arr, dep_time, arr_time, aircraft_type, min_turn_time)
    return Leg(;
        id="L$(lpad(index, 4, '0'))",
        departure_airport=dep,
        arrival_airport=arr,
        departure_time=dep_time,
        arrival_time=arr_time,
        aircraft_type,
        minimum_turn_time=min_turn_time,
    )
end

"""
$(TYPEDSIGNATURES)

Generate a single aircraft rotation (chain of legs starting and mostly returning to `hub`)
in place into `legs`, stopping early (rather than skipping to a fresh hub departure) as
soon as the next candidate leg would arrive after `horizon_end`.

Returns the `(airport, time)` tail state of the rotation (last arrival airport and time)
so that [`_pad_to_target!`](@ref) can later extend it, or `nothing` if no leg could be
generated at all (e.g. the rotation's random start time leaves no room before the
horizon).
"""
function _generate_rotation!(
    legs,
    leg_counter,
    rng,
    num_legs;
    hub,
    spokes,
    spoke_distances,
    hub_fraction,
    horizon_start,
    horizon_end,
    horizon_minutes,
    min_flight_duration,
    max_flight_duration,
    min_turn_time,
    max_turn_time,
    tight_turn_fraction,
    aircraft_type,
)
    current_airport = hub
    current_time = horizon_start + Dates.Minute(rand(rng, 0:(horizon_minutes ÷ 3)))
    produced_any = false

    for _ in 1:num_legs
        dest = _pick_destination(rng, current_airport, hub, spokes, hub_fraction)
        arr_time = _compute_arrival(
            rng,
            current_time,
            current_airport,
            dest,
            hub,
            spoke_distances,
            min_flight_duration,
            max_flight_duration,
        )

        # Stop the rotation rather than jumping back to the hub with a fresh start time:
        # a fresh start would create a leg chain that no single aircraft can fly, since the
        # legs already pushed for this rotation are still attributed to the same aircraft.
        arr_time > horizon_end && break

        leg_counter[] += 1
        push!(
            legs,
            _make_leg(
                leg_counter[],
                current_airport,
                dest,
                current_time,
                arr_time,
                aircraft_type,
                min_turn_time,
            ),
        )

        turn = _sample_turn_time(rng, min_turn_time, max_turn_time, tight_turn_fraction)
        current_airport = dest
        current_time = arr_time + Dates.Minute(turn)
        produced_any = true
    end

    return produced_any ? (current_airport, current_time) : nothing
end

function _compute_arrival(
    rng, dep_time, dep_airport, arr_airport, hub, spoke_distances, min_dur, max_dur
)
    dist = _flight_distance(dep_airport, arr_airport, hub, spoke_distances)
    duration = _sample_duration(rng, dist, min_dur, max_dur)
    return dep_time + Dates.Minute(duration)
end

"""
$(TYPEDSIGNATURES)

Pad `legs` up to `target` legs by appending legs to the tail of existing rotations
(described by `rotation_tails`, a vector of `(airport, time)` states as returned by
[`_generate_rotation!`](@ref)), instead of creating brand new hub departures that would
require hidden extra aircraft.

At each step, the rotation with the earliest last arrival time is extended first (from its
current airport, after a random turn time, to the hub or a spoke). A rotation that cannot
take another leg before `horizon_end` is dropped from consideration. If no rotation can
take more legs, padding stops early and `length(legs)` stays below `target`: callers must
tolerate a leg count slightly below the requested one in that case.
"""
function _pad_to_target!(
    legs,
    leg_counter,
    rotation_tails::Vector{Tuple{String,Dates.DateTime}},
    rng,
    target;
    hub,
    spokes,
    spoke_distances,
    hub_fraction,
    horizon_end,
    min_flight_duration,
    max_flight_duration,
    min_turn_time,
    max_turn_time,
    tight_turn_fraction,
    aircraft_type,
)
    active = collect(eachindex(rotation_tails))

    while length(legs) < target && !isempty(active)
        best = argmin(i -> rotation_tails[i][2], active)
        current_airport, current_time = rotation_tails[best]

        dest = _pick_destination(rng, current_airport, hub, spokes, hub_fraction)
        arr_time = _compute_arrival(
            rng,
            current_time,
            current_airport,
            dest,
            hub,
            spoke_distances,
            min_flight_duration,
            max_flight_duration,
        )

        if arr_time > horizon_end
            filter!(!=(best), active)
            continue
        end

        leg_counter[] += 1
        push!(
            legs,
            _make_leg(
                leg_counter[],
                current_airport,
                dest,
                current_time,
                arr_time,
                aircraft_type,
                min_turn_time,
            ),
        )

        turn = _sample_turn_time(rng, min_turn_time, max_turn_time, tight_turn_fraction)
        rotation_tails[best] = (dest, arr_time + Dates.Minute(turn))
    end
end

function _finalize_legs(legs::Vector{Leg}, nb_legs::Int)
    result = legs[1:min(nb_legs, length(legs))]
    sort!(result; by=departure_time)
    for i in eachindex(result)
        result[i] = _make_leg(
            i,
            result[i].departure_airport,
            result[i].arrival_airport,
            departure_time(result[i]),
            arrival_time(result[i]),
            result[i].aircraft_type,
            result[i].minimum_turn_time,
        )
    end
    return result
end

function _pick_destination(
    rng, current::String, hub::String, spokes::Vector{String}, hub_fraction::Float64
)
    if current == hub
        return spokes[rand(rng, 1:length(spokes))]
    end
    if rand(rng) < hub_fraction
        return hub
    end
    candidates = filter(!=(current), spokes)
    isempty(candidates) && return hub
    return candidates[rand(rng, 1:length(candidates))]
end

function _flight_distance(
    dep::String, arr::String, hub::String, spoke_distances::Dict{String,Float64}
)
    dep == hub && return spoke_distances[arr]
    arr == hub && return spoke_distances[dep]
    return (get(spoke_distances, dep, 0.5) + get(spoke_distances, arr, 0.5)) * 0.4
end

function _sample_duration(rng, distance::Float64, min_dur::Int, max_dur::Int)
    base = min_dur + round(Int, distance * (max_dur - min_dur))
    return clamp(base + rand(rng, -30:30), min_dur, max_dur)
end

"""
$(TYPEDSIGNATURES)

Compute the minimum number of aircraft needed to cover all of `legs`, i.e. the minimum
path cover of the leg chaining DAG (an edge `u -> v` exists whenever `v` could
immediately follow `u` on the same aircraft, per [`is_valid_chaining`](@ref)).

By König's theorem, the minimum path cover of a DAG equals the number of vertices minus
the size of a maximum matching of the bipartite graph obtained by splitting each vertex
into an "out" and an "in" copy. The matching is computed with a simple Kuhn
augmenting-path algorithm, which is fast enough for the instance sizes generated here.
"""
function minimum_fleet_size(legs::Vector{<:AbstractLeg}; TTM_factor::Float64=0.0)
    L = length(legs)
    successors = [Int[] for _ in 1:L]
    for u in 1:L, v in 1:L
        if u != v && is_valid_chaining(legs[u], legs[v]; TTM_factor)
            push!(successors[u], v)
        end
    end

    match_of_successor = zeros(Int, L)

    function try_augment!(u::Int, visited::BitVector)
        for v in successors[u]
            visited[v] && continue
            visited[v] = true
            if match_of_successor[v] == 0 || try_augment!(match_of_successor[v], visited)
                match_of_successor[v] = u
                return true
            end
        end
        return false
    end

    matching_size = 0
    for u in 1:L
        visited = falses(L)
        matching_size += try_augment!(u, visited)
    end

    return L - matching_size
end

"""
$(TYPEDSIGNATURES)

Compute the minimum number of aircraft needed to cover all legs of `schedule`.
See [`minimum_fleet_size(::Vector{<:AbstractLeg})`](@ref).
"""
function minimum_fleet_size(schedule::ActivitySchedule)
    return minimum_fleet_size(schedule.legs; TTM_factor=schedule.TTM_factor)
end
