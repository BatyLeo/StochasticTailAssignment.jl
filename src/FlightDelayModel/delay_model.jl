"""
$TYPEDEF

Synthetic delay model: a [`HandcraftedDelayPredictor`](@ref) together with the
shift, clamp and intensity parameters turning its LogNormal/Normal parameter
predictions into intrinsic (root) departure and arrival delays.

[`has_dynamic_features`](@ref) reports whether any beta coefficient is nonzero, and if
so only [`propagate_delays`](@ref) may be used.

# Fields
$TYPEDFIELDS
"""
@kwdef struct SyntheticDelayModel{P}
    "predictor mapping static features to LogNormal/Normal distribution parameters"
    predictor::P
    "positive shift of the root departure LogNormal (minutes)"
    dep_shift::Float64 = 21.0
    "positive shift of the root arrival Normal (minutes)"
    arr_shift::Float64 = 42.0
    "maximum departure root delay allowed (minutes)"
    max_dep::Float64 = 78.0
    "maximum arrival root delay allowed (minutes)"
    max_arr::Float64 = 23.0
    "global multiplier applied to every root delay after shift and cap"
    delay_intensity::Float64 = 1.0
end

"""
$TYPEDSIGNATURES

Check whether `model` uses dynamic (route-dependent) features, i.e. whether
any of `beta_dep_has_propagated`, `beta_dep_propagated`,
`beta_arr_has_departure_delay`, `beta_arr_propagated` in `model.predictor.coefficients`
is nonzero (see [`SyntheticDelayModel`](@ref) and [`DelayCoefficients`](@ref)).
"""
function has_dynamic_features(model::SyntheticDelayModel)
    c = model.predictor.coefficients
    return !iszero(c.beta_dep_has_propagated) ||
           !iszero(c.beta_dep_propagated) ||
           !iszero(c.beta_arr_has_departure_delay) ||
           !iszero(c.beta_arr_propagated)
end

"""
$TYPEDSIGNATURES

Check that `scenarios` and `model.predictor` were built from the same
`FeaturesConfig` (same feature names, in the same order), throwing a clear
error otherwise.

A mismatch (e.g. `DelayScenarios` built with the default airport list while
`model` was built with `schedule_airports(schedule)`, or the reverse) silently
misaligns every feature-weight dot product, so this should always be checked
before combining a `DelayScenarios` and a `SyntheticDelayModel`.
"""
function _check_feature_layout(scenarios::DelayScenarios, model::SyntheticDelayModel)
    scenario_names = scenarios.feature_names
    predictor_names = model.predictor.feature_names
    scenario_names == predictor_names || throw(
        ArgumentError(
            "feature layout mismatch between `scenarios` ($(length(scenario_names)) " *
            "features) and `model.predictor` ($(length(predictor_names)) features): " *
            "`DelayScenarios` and `SyntheticDelayModel` must be built from the same " *
            "`FeaturesConfig` (e.g. both using `schedule_airports(schedule)`)",
        ),
    )
    return nothing
end

"""
$TYPEDSIGNATURES

Sample the intrinsic (root) departure and arrival delays of leg `leg_index`,
for every scenario in `scenarios`, as `(eps_dep, eps_arr)` pairs.

!!! warning
    `model` should only use static features: `has_dynamic_features(model)` must
    be `false`. Dynamic (route-dependent) delay effects are only supported by
    [`propagate_delays`](@ref).
"""
function sample_root_scenario(
    model::SyntheticDelayModel, scenarios::DelayScenarios, leg_index::Int
)
    _check_feature_layout(scenarios, model)
    (;
        departure_scenarios,
        arrival_scenarios,
        departure_reparameterize,
        arrival_reparameterize,
    ) = scenarios
    (; dep_shift, arr_shift, max_dep, max_arr, delay_intensity) = model

    x_dep = feature_vector_departure(scenarios, leg_index)
    x_arr = feature_vector_arrival(scenarios, leg_index)
    (; mu_dep, sigma_dep) = departure_prediction(model.predictor, x_dep)
    (; mu_arr, sigma_arr) = arrival_prediction(model.predictor, x_arr)

    return map(1:nb_scenarios(scenarios)) do s
        eps_dep =
            min(
                departure_reparameterize(
                    mu_dep, sigma_dep, departure_scenarios[leg_index, s], dep_shift
                ),
                max_dep,
            ) * delay_intensity
        eps_arr =
            min(
                arrival_reparameterize(
                    mu_arr, sigma_arr, arrival_scenarios[leg_index, s], arr_shift
                ),
                max_arr,
            ) * delay_intensity
        return (eps_dep, eps_arr)
    end
end

"""
$TYPEDSIGNATURES

Sum of the departure and arrival intrinsic delays of leg `leg_index`, for every
scenario in `scenarios`.
"""
function sample_root_scenario_sum(
    model::SyntheticDelayModel, scenarios::DelayScenarios, leg_index::Int
)
    return map(sample_root_scenario(model, scenarios, leg_index)) do (eps_dep, eps_arr)
        return eps_dep + eps_arr
    end
end

"""
$TYPEDSIGNATURES

Sample root delay scenarios for `model` and `scenarios`, summing departure and
arrival intrinsic delays into a single value per leg and scenario.

Returns a `nb_scenarios(scenarios) x nb_legs(scenarios)` `Matrix{Float32}`.

!!! warning
    `model` should only use static features: see [`sample_root_scenario`](@ref).
"""
function sample_root_scenarios(model::SyntheticDelayModel, scenarios::DelayScenarios)
    has_dynamic_features(model) &&
        throw(ArgumentError("sample_root_scenarios requires a static-only model"))
    res = [sample_root_scenario_sum(model, scenarios, l) for l in 1:nb_legs(scenarios)]
    return Float32.(hcat(res...))
end

"""
$TYPEDSIGNATURES

Sample root delay scenarios for `model` and `scenarios`, without summing the
departure and arrival intrinsic delays.

Returns a named tuple `(; departure, arrival)` of two
`nb_scenarios(scenarios) x nb_legs(scenarios)` `Matrix{Float32}`.

!!! warning
    `model` should only use static features: see [`sample_root_scenario`](@ref).
"""
function sample_root_scenarios_unmerged(
    model::SyntheticDelayModel, scenarios::DelayScenarios
)
    has_dynamic_features(model) &&
        throw(ArgumentError("sample_root_scenarios_unmerged requires a static-only model"))
    L = nb_legs(scenarios)
    S = nb_scenarios(scenarios)
    departure = zeros(Float32, S, L)
    arrival = zeros(Float32, S, L)
    for l in 1:L
        root = sample_root_scenario(model, scenarios, l)
        for s in 1:S
            departure[s, l], arrival[s, l] = root[s]
        end
    end
    return (; departure, arrival)
end
