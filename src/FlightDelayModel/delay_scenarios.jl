"""
$TYPEDEF

Precomputed standard normal scenarios for a given schedule, used through the
reparameterization trick to sample intrinsic delays.

# Fields
$TYPEDFIELDS
"""
struct DelayScenarios{T,R1,R2,F}
    "associated schedule to store scenarios on"
    schedule::ActivitySchedule
    "matrix of precomputed standard normal draws for departure, size `nb_legs x nb_scenarios`"
    departure_scenarios::Matrix{T}
    "matrix of precomputed standard normal draws for arrival, size `nb_legs x nb_scenarios`"
    arrival_scenarios::Matrix{T}
    "method transforming a scenario draw and distribution parameters into a departure intrinsic delay"
    departure_reparameterize::R1
    "method transforming a scenario draw and distribution parameters into an arrival intrinsic delay"
    arrival_reparameterize::R2
    "static features (shared between departure and arrival), size `nb_features x nb_legs`"
    static_features::F
    "names of the static features (from the `FeaturesConfig` used to build the scenarios), used to check consistency against a `SyntheticDelayModel`'s predictor"
    feature_names::Vector{String}
end

function Base.getindex(scenarios::DelayScenarios, idx...)
    return DelayScenarios(
        scenarios.schedule,
        scenarios.departure_scenarios[:, idx...],
        scenarios.arrival_scenarios[:, idx...],
        scenarios.departure_reparameterize,
        scenarios.arrival_reparameterize,
        scenarios.static_features,
        scenarios.feature_names,
    )
end

function Base.getindex(scenarios::DelayScenarios, idx::Integer)
    return scenarios[idx:idx]
end

function nb_scenarios(scenarios::DelayScenarios)
    return size(scenarios.departure_scenarios, 2)
end
Base.length(scenarios::DelayScenarios) = nb_scenarios(scenarios)

"""
$TYPEDSIGNATURES

Compute the reparameterized intrinsic delay for `scenario_value` with a
lognormal distribution.
"""
function lognormal_reparameterize(mu, sigma, scenario_value; round_to_int=true)
    value = exp(mu + sigma * scenario_value)
    return round_to_int ? round(value) : value
end

"""
$TYPEDSIGNATURES

Compute the reparameterized intrinsic delay for `scenario_value` with a
gaussian distribution.
"""
function normal_reparameterize(mu, sigma, scenario_value; round_to_int=true)
    value = mu + sigma * scenario_value
    return round_to_int ? round(value) : value
end

"""
$TYPEDSIGNATURES

Compute the shifted, reparameterized intrinsic delay for `scenario_value` with
a lognormal distribution.
"""
function lognormal_reparameterize(mu, sigma, scenario_value, shift; round_to_int=true)
    value = exp(mu + sigma * scenario_value) - shift
    return round_to_int ? round(value) : value
end

"""
$TYPEDSIGNATURES

Compute the shifted, reparameterized intrinsic delay for `scenario_value` with
a gaussian distribution.
"""
function normal_reparameterize(mu, sigma, scenario_value, shift; round_to_int=true)
    value = mu + sigma * scenario_value - shift
    return round_to_int ? round(value) : value
end

"""
$TYPEDSIGNATURES

Custom constructor for [`DelayScenarios`](@ref): draws `nb_scenarios` standard
normal scenarios per leg of `schedule`, column by column, from two dedicated
`Random.Xoshiro` random number generators, seeded respectively from
`hash((seed, :departure))` and `hash((seed, :arrival))` (so that increasing
`nb_scenarios` never changes the earlier columns, and so that the arrival
stream of a given `seed` never collides with the departure stream of another
`seed`), and computes the static features of every leg using `config`.
"""
function DelayScenarios(
    schedule::ActivitySchedule;
    nb_scenarios::Int=1,
    departure_reparameterize=lognormal_reparameterize,
    arrival_reparameterize=normal_reparameterize,
    config::FeaturesConfig=FeaturesConfig(; airports=schedule_airports(schedule)),
    seed::Int=0,
)
    rng_dep = Random.Xoshiro(hash((seed, :departure)))
    rng_arr = Random.Xoshiro(hash((seed, :arrival)))
    L = nb_legs(schedule)

    static_features = compute_leg_features(schedule, config)
    departure_scenarios = Float32[randn(rng_dep) for _ in 1:L, _ in 1:nb_scenarios]
    arrival_scenarios = Float32[randn(rng_arr) for _ in 1:L, _ in 1:nb_scenarios]

    return DelayScenarios(
        schedule,
        departure_scenarios,
        arrival_scenarios,
        departure_reparameterize,
        arrival_reparameterize,
        static_features,
        config.names,
    )
end

AircraftRoutingBase.nb_legs(scenarios::DelayScenarios) = nb_legs(scenarios.schedule)

"""
$TYPEDSIGNATURES

Retrieve the static feature vector associated to leg index `l`, for departure.
"""
function feature_vector_departure(scenarios::DelayScenarios, l::Int)
    return @view scenarios.static_features[:, l]
end

"""
$TYPEDSIGNATURES

Retrieve the static feature vector associated to leg index `l`, for arrival.
"""
function feature_vector_arrival(scenarios::DelayScenarios, l::Int)
    return @view scenarios.static_features[:, l]
end
