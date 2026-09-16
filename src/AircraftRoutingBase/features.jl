"""
Default per-airport size lookup, in `[0, 1]`, for the synthetic airport codes
used by the instance generator (`HUB` is the hub, `A` to `G` are spokes).
"""
const DEFAULT_AIRPORT_SIZES = Dict(
    "HUB" => 1.0,
    "A" => 0.7,
    "B" => 0.6,
    "C" => 0.55,
    "D" => 0.5,
    "E" => 0.4,
    "F" => 0.3,
    "G" => 0.2,
)

"""
Fallback airport size used for airports not present in `DEFAULT_AIRPORT_SIZES`
(or in a custom `airport_sizes` dictionary).
"""
const DEFAULT_AIRPORT_SIZE = 0.4

"""
$TYPEDEF

Configuration for computing static leg features.

This does not carry any min-max scaling information, the handcrafted delay model
works directly on the raw, unscaled feature values.

# Fields
$TYPEDFIELDS
"""
@kwdef struct FeaturesConfig{A}
    "list of available airports (categories other than these are grouped as `other`)"
    airports::Vector{String} = ["HUB", "A", "B", "C", "D", "E", "F", "G"]
    "list of all possible days of week (1 = Monday, ..., 7 = Sunday)"
    days::Vector{Int} = collect(1:7)
    "list of all possible departure hours"
    start_hours::Vector{Int} = collect(0:23)
    "list of all possible arrival hours"
    end_hours::Vector{Int} = collect(0:23)
    "store info about airport sizes, in [0, 1]"
    airport_sizes::A = DEFAULT_AIRPORT_SIZES
    "list of feature names (output of `compute_leg_feature_names`)"
    names::Vector{String} = compute_leg_feature_names(;
        airports, days, start_hours, end_hours
    )
end

"""
$TYPEDSIGNATURES

Compute the airport size of the departure airport of `leg`, defaulting to
`DEFAULT_AIRPORT_SIZE` for airports not present in `airport_sizes`.
"""
function leg_airport_size(leg::AbstractActivity, airport_sizes::AbstractDict)
    return get(airport_sizes, departure_airport(leg), DEFAULT_AIRPORT_SIZE)
end

"""
$TYPEDSIGNATURES

Compute the static features of the input activity, using the given configuration.

Let `A = length(airports)` and `C = A + 1` (the airport categories, plus the `other`
category). Concatenate the following features, in order:
1. minimum turn time, in hours
2. duration of the leg, in hours
3-9. one-hot encoding of the day of week (7 values)
10-33. one-hot encoding of the departure hour (24 values)
34-57. one-hot encoding of the arrival hour (24 values)
58 to (57 + C). one-hot encoding of the departure airport, including an `other` category
(`C` values)
(58 + C) to (57 + 2C). one-hot encoding of the arrival airport, including an `other`
category (`C` values)
(58 + 2C). airport size of the departure airport, in `[0, 1]`
(59 + 2C). flight distance proxy, `clamp(duration_hours / 10, 0, 1)`
(60 + 2C). haul type, `1` if the leg duration is more than 240 minutes, `0` otherwise
"""
function compute_leg_features(
    leg::AbstractActivity;
    airports::AbstractVector{String},
    days::AbstractVector{Int}=collect(1:7),
    start_hours::AbstractVector{Int}=collect(0:23),
    end_hours::AbstractVector{Int}=collect(0:23),
    airport_sizes::AbstractDict=DEFAULT_AIRPORT_SIZES,
    type=Float32,
)
    duration_minutes = duration(leg)
    duration_hours = duration_minutes / 60
    res = vcat(
        minimum_turn_time(leg) / 60,
        duration_hours,
        onehot_dayofweek(leg, days),
        onehot_departure_hour(leg, start_hours),
        onehot_arrival_hour(leg, end_hours),
        onehot_departure_airport(leg, airports),
        onehot_arrival_airport(leg, airports),
        leg_airport_size(leg, airport_sizes),
        clamp(duration_hours / 10, 0, 1),
        duration_minutes > 240 ? 1 : 0,
    )
    return type.(res)
end

"""
$TYPEDSIGNATURES

Compute the static feature names, in the same order as [`compute_leg_features`](@ref).
"""
function compute_leg_feature_names(;
    airports::AbstractVector{String},
    days::AbstractVector{Int}=collect(1:7),
    start_hours::AbstractVector{Int}=collect(0:23),
    end_hours::AbstractVector{Int}=collect(0:23),
)
    return vcat(
        ["TTM", "Duration"],
        ["DayOfWeek/$i" for i in days],
        ["DepartureHour/$i" for i in start_hours],
        ["ArrivalHour/$i" for i in end_hours],
        vcat(["DepartureAirport/$i" for i in airports], "DepartureAirport/other"),
        vcat(["ArrivalAirport/$i" for i in airports], "ArrivalAirport/other"),
        ["AirportSize", "FlightDistance", "HaulType"],
    )
end

function compute_leg_features(leg::AbstractActivity, config::FeaturesConfig; type=Float32)
    (; airports, days, start_hours, end_hours, airport_sizes) = config
    return compute_leg_features(
        leg; airports, days, start_hours, end_hours, airport_sizes, type
    )
end

function compute_leg_features(
    legs::Vector{<:AbstractActivity}, config::FeaturesConfig; type=Float32, kwargs...
)
    n = length(config.names)
    X = Matrix{type}(undef, n, length(legs))
    for (i, leg) in enumerate(legs)
        X[:, i] .= compute_leg_features(leg, config; type, kwargs...)
    end
    return X
end

function compute_leg_features(schedule::ActivitySchedule, config::FeaturesConfig; kwargs...)
    return compute_leg_features(schedule.legs, config; kwargs...)
end

"""
$TYPEDSIGNATURES

Compute lookup information associated to `config`: the feature names, and a
`name => index` dictionary giving the position of each named feature in the
vectors returned by [`compute_leg_features`](@ref).
"""
function compute_feature_info(config::FeaturesConfig)
    names = config.names
    index = Dict(name => i for (i, name) in enumerate(names))
    return (; names, index)
end

"""
$(TYPEDSIGNATURES)

List the airports used by `schedule`, ordered by decreasing usage (as either
departure or arrival airport), so that `first(...)` is the busiest one (the
hub, for a schedule generated by `generate_schedule`). Airports with equal usage counts
are ordered alphabetically, so the result is a deterministic total order.

Used to build a [`FeaturesConfig`](@ref) that matches the airport codes
actually present in `schedule`, rather than assuming the `["HUB", "A", "B",
"C", "D", "E", "F", "G"]` default, which only matches `generate_schedule`'s
own default when explicitly requested (e.g. via `generate_benchmark_instance`).

Defined here (rather than in `InstanceGenerator`) so that both
`FlightDelayModel` (default `config` of [`DelayScenarios`](@ref)) and
`InstanceGenerator` (default `config` of `build_delay_model`) can build a
matching, consistent [`FeaturesConfig`](@ref) without a circular dependency.
"""
function schedule_airports(schedule::ActivitySchedule)
    counts = Dict{String,Int}()
    for leg in schedule.legs
        counts[departure_airport(leg)] = get(counts, departure_airport(leg), 0) + 1
        counts[arrival_airport(leg)] = get(counts, arrival_airport(leg), 0) + 1
    end
    return sort(collect(keys(counts)); by=a -> (-counts[a], a))
end
