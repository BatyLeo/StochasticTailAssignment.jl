function _assign_spoke_distances(rng, spokes::Vector{String})
    return Dict(sp => (i - 1 + rand(rng)) / length(spokes) for (i, sp) in enumerate(spokes))
end

function _allocate_legs(rng, nb_legs::Int, nb_aircraft::Int)
    weights = [0.3 + rand(rng) for _ in 1:nb_aircraft]
    allocation = [max(1, round(Int, nb_legs * w / sum(weights))) for w in weights]
    diff = nb_legs - sum(allocation)
    for i in 1:abs(diff)
        allocation[((i-1)%nb_aircraft)+1] += sign(diff)
    end
    return allocation
end

function _sample_turn_time(rng, min_turn::Int, max_turn::Int, tight_fraction::Float64)
    if rand(rng) < tight_fraction && min_turn > 1
        return rand(rng, max(1, min_turn÷2):(min_turn-1))
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
    current_time = horizon_start + Dates.Minute(rand(rng, 0:(horizon_minutes÷3)))

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

        if arr_time > horizon_end
            current_airport = hub
            current_time = horizon_start + Dates.Minute(rand(rng, 0:(horizon_minutes÷3)))
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
            arr_time > horizon_end && break
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
        current_airport = dest
        current_time = arr_time + Dates.Minute(turn)
    end
end

function _compute_arrival(
    rng, dep_time, dep_airport, arr_airport, hub, spoke_distances, min_dur, max_dur
)
    dist = _flight_distance(dep_airport, arr_airport, hub, spoke_distances)
    duration = _sample_duration(rng, dist, min_dur, max_dur)
    return dep_time + Dates.Minute(duration)
end

function _pad_to_target!(
    legs,
    leg_counter,
    rng,
    target;
    hub,
    spokes,
    spoke_distances,
    horizon_start,
    horizon_end,
    horizon_minutes,
    min_flight_duration,
    max_flight_duration,
    aircraft_type,
    min_turn_time,
)
    while length(legs) < target
        dep_time = horizon_start + Dates.Minute(rand(rng, 0:(horizon_minutes÷2)))
        dest = spokes[rand(rng, 1:length(spokes))]
        arr_time = _compute_arrival(
            rng,
            dep_time,
            hub,
            dest,
            hub,
            spoke_distances,
            min_flight_duration,
            max_flight_duration,
        )
        arr_time > horizon_end && continue

        leg_counter[] += 1
        push!(
            legs,
            _make_leg(
                leg_counter[], hub, dest, dep_time, arr_time, aircraft_type, min_turn_time
            ),
        )
    end
end

function _finalize_legs(legs::Vector{Leg}, nb_legs::Int)
    result = legs[1:nb_legs]
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
