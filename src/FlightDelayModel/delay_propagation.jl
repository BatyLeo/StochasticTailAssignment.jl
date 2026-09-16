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

"""
$TYPEDSIGNATURES

Simulate a single scenario `s` of `model` on `route`, threading the upstream
propagated delay through the dynamic (route-dependent) departure and arrival
predictions (see [`DelayCoefficients`](@ref)`.beta_dep_has_propagated` and
friends).

Returns a named tuple `(; departure_delays, arrival_delays, departure_root_delays,
arrival_root_delays, propagated_delays)`, each a `Vector{Float32}` of length
`length(route)` indexed by route position (maintenance positions hold the
carried-over propagated delay, with a `0` intrinsic delay).

!!! warning
    If any `beta_*` coefficient of `model.predictor.coefficients` is nonzero,
    the resulting delays are route-dependent and must not be reused as input to
    `stochastic_column_generation` (unlike [`sample_root_scenarios`](@ref)).
"""
function propagate_delays(
    model::SyntheticDelayModel, scenarios::DelayScenarios, route::Route, s::Int
)
    _check_feature_layout(scenarios, model)
    schedule = scenarios.schedule
    (;
        departure_scenarios,
        arrival_scenarios,
        departure_reparameterize,
        arrival_reparameterize,
    ) = scenarios
    (; dep_shift, arr_shift, max_dep, max_arr, delay_intensity) = model
    c = model.predictor.coefficients

    R = length(route)
    departure_delays = zeros(Float32, R)
    arrival_delays = zeros(Float32, R)
    departure_root_delays = zeros(Float32, R)
    arrival_root_delays = zeros(Float32, R)
    propagated_delays = zeros(Float32, R)

    propagated_delay = 0.0f0
    for (index, i) in enumerate(route)
        if is_maintenance(schedule, i)
            xi_dep = propagated_delay
            xi_arr = propagated_delay
            eps_dep = 0.0f0
            eps_arr = 0.0f0
        else
            x_dep = feature_vector_departure(scenarios, i)
            has_prop = propagated_delay > 0 ? 1.0 : 0.0
            mu_offset_dep =
                c.beta_dep_has_propagated * has_prop +
                c.beta_dep_propagated * (propagated_delay / 60)
            (; mu_dep, sigma_dep) = departure_prediction(
                model.predictor, x_dep; mu_offset=mu_offset_dep
            )
            eps_dep =
                min(
                    departure_reparameterize(
                        mu_dep, sigma_dep, departure_scenarios[i, s], dep_shift
                    ),
                    max_dep,
                ) * delay_intensity
            xi_dep = propagated_delay + eps_dep

            x_arr = feature_vector_arrival(scenarios, i)
            has_dep = xi_dep > 0 ? 1.0 : 0.0
            mu_offset_arr =
                c.beta_arr_has_departure_delay * has_dep +
                c.beta_arr_propagated * (xi_dep / 60)
            (; mu_arr, sigma_arr) = arrival_prediction(
                model.predictor, x_arr; mu_offset=mu_offset_arr
            )
            eps_arr =
                min(
                    arrival_reparameterize(
                        mu_arr, sigma_arr, arrival_scenarios[i, s], arr_shift
                    ),
                    max_arr,
                ) * delay_intensity
            xi_arr = xi_dep + eps_arr
        end

        departure_delays[index] = xi_dep
        arrival_delays[index] = xi_arr
        departure_root_delays[index] = eps_dep
        arrival_root_delays[index] = eps_arr

        if index < R
            activity = get_activity(schedule, i)
            next_activity = get_activity(schedule, route[index + 1])
            propagated_delay = max(
                xi_arr - slack_with_turn_time(activity, next_activity), 0.0f0
            )
            propagated_delays[index + 1] = propagated_delay
        end
    end

    return (;
        departure_delays,
        arrival_delays,
        departure_root_delays,
        arrival_root_delays,
        propagated_delays,
    )
end

"""
$TYPEDSIGNATURES

Simulate every scenario of `model` on `routes`, for every leg of `scenarios`'s
schedule (see [`propagate_delays`](@ref) for a single route and scenario).

Returns a named tuple `(; departure_delays, arrival_delays, departure_root_delays,
arrival_root_delays, propagated_delays)`, each a `nb_scenarios(scenarios) x
nb_legs(scenarios)` `Matrix{Float32}`, directly comparable to the output of
[`propagate_delays_from_root_delays`](@ref) when every `beta_*` coefficient of
`model.predictor.coefficients` is `0`.
"""
function propagate_delays(
    model::SyntheticDelayModel, scenarios::DelayScenarios, routes::Vector{Route}
)
    schedule = scenarios.schedule
    L = nb_legs(scenarios)
    S = nb_scenarios(scenarios)

    departure_delays = zeros(Float32, S, L)
    arrival_delays = zeros(Float32, S, L)
    departure_root_delays = zeros(Float32, S, L)
    arrival_root_delays = zeros(Float32, S, L)
    propagated_delays = zeros(Float32, S, L)

    for route in routes, s in 1:S
        route_delays = propagate_delays(model, scenarios, route, s)
        for (index, i) in enumerate(route)
            is_maintenance(schedule, i) && continue
            departure_delays[s, i] = route_delays.departure_delays[index]
            arrival_delays[s, i] = route_delays.arrival_delays[index]
            departure_root_delays[s, i] = route_delays.departure_root_delays[index]
            arrival_root_delays[s, i] = route_delays.arrival_root_delays[index]
            propagated_delays[s, i] = route_delays.propagated_delays[index]
        end
    end

    return (;
        departure_delays,
        arrival_delays,
        departure_root_delays,
        arrival_root_delays,
        propagated_delays,
    )
end
