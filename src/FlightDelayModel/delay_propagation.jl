"""
$TYPEDSIGNATURES

Propagate input `root_delays` to compute delays for each leg in `route`.
"""
function propagate_delays_from_root_delays(
    route::Route, root_delays::AbstractMatrix, schedule::ActivitySchedule
)
    S = size(root_delays, 1)

    arrival_delays = zeros(Float32, S, length(route))
    propagated_delay = zeros(Float32, S)

    for (index, i) in enumerate(route)
        arrival_delay =
            propagated_delay .+
            (is_maintenance(schedule, i) ? 0.0 : @view(root_delays[:, i]))
        if index < length(route)
            activity = get_activity(schedule, i)
            next_activity = get_activity(schedule, route[index + 1])
            propagated_delay .= max.(
                arrival_delay .- slack_with_turn_time(activity, next_activity), 0
            )
        end
        if !is_maintenance(schedule, i)
            arrival_delays[:, index] .= arrival_delay
        end
    end
    return arrival_delays
end

"""
$TYPEDSIGNATURES

Propagate input `root_delays` to compute delays for each leg in `routes`.
"""
function propagate_delays_from_root_delays(
    routes::AbstractVector{Route}, root_delays::AbstractMatrix, schedule::ActivitySchedule
)
    L = nb_legs(schedule)
    S = size(root_delays, 1)
    @assert size(root_delays, 2) == L

    arrival_delays = zeros(Float32, S, L)
    for route in routes
        propagated_delay = zeros(Float32, S)
        for (index, i) in enumerate(route)
            arrival_delay =
                propagated_delay .+ (is_maintenance(schedule, i) ? 0.0 : root_delays[:, i])
            if index < length(route)
                activity = get_activity(schedule, i)
                next_activity = get_activity(schedule, route[index + 1])
                propagated_delay .= max.(
                    arrival_delay .- slack_with_turn_time(activity, next_activity), 0
                )
            end
            if !is_maintenance(schedule, i)
                arrival_delays[:, i] .= arrival_delay
            end
        end
    end
    return arrival_delays
end
