"""
$TYPEDEF

Abstract type defining an activity.

Two implementations
- [`Leg`](@ref)
- [`Maintenance`](@ref)

# Common methods
- [`is_maintenance`](@ref)
- [`start_airport`](@ref)
- [`end_airport`](@ref)
- [`start_time`](@ref)
- [`end_time`](@ref)
"""
abstract type AbstractActivity end

"""
$TYPEDSIGNATURES

Compute the duration of an activity in minutes.
"""
duration(activity::AbstractActivity) = to_minutes(end_time(activity) - start_time(activity))

"""
$TYPEDSIGNATURES

Retrieve the minimum turn time for an activity.
"""
minimum_turn_time(activity::AbstractActivity) = activity.minimum_turn_time

"""
$TYPEDSIGNATURES

Retrieve the id of given activity.
"""
id(activity::AbstractActivity) = activity.id

"""
$TYPEDSIGNATURES

Retrieve the day of week of `activity` start time.
"""
function Dates.dayofweek(activity::AbstractActivity)
    return Dates.dayofweek(start_time(activity))
end

"""
$TYPEDSIGNATURES

Encode the day of week of given activity as a one-hot vector.
"""
function onehot_dayofweek(activity::AbstractActivity, range=1:7)
    return onehot(Dates.dayofweek(activity), vcat(range, -1), -1)[1:(end - 1)]
end

"""
$TYPEDSIGNATURES

Retrieve the hour of `activity` start time.
"""
function start_hour(activity::AbstractActivity)
    return Dates.hour(start_time(activity))
end

"""
$TYPEDSIGNATURES

Alias for [`start_hour`](@ref).
"""
const departure_hour = start_hour

"""
$TYPEDSIGNATURES

Retrieve the hour of `activity` start time.
"""
function end_hour(activity::AbstractActivity)
    return Dates.hour(end_time(activity))
end

"""
$TYPEDSIGNATURES

Alias for [`end_hour`](@ref).
"""
const arrival_hour = end_hour

"""
$TYPEDSIGNATURES

Encode the start hour of given activity as a one-hot vector.
"""
function onehot_start_hour(activity::AbstractActivity, range=0:23)
    return onehot(start_hour(activity), vcat(range, -1), -1)[1:(end - 1)]
end

"""
$TYPEDSIGNATURES

Alias for [`onehot_start_hour`](@ref).
"""
const onehot_departure_hour = onehot_start_hour

"""
$TYPEDSIGNATURES

Encode the and hour of given activity as a one-hot vector.
"""
function onehot_end_hour(activity::AbstractActivity, range=0:23)
    return onehot(end_hour(activity), vcat(range, -1), -1)[1:(end - 1)]
end

"""
$TYPEDSIGNATURES

Alias for [`onehot_end_hour`](@ref).
"""
const onehot_arrival_hour = onehot_end_hour

"""
$TYPEDSIGNATURES

Retrieve the month of `activity` start time.
"""
function Dates.month(activity::AbstractActivity)
    return Dates.month(start_time(activity))
end

"""
$TYPEDSIGNATURES

Encode the start month of given activity as a one-hot vector.
"""
function onehot_month(activity::AbstractActivity, range=1:12)
    return onehot(Dates.month(activity), vcat(range, -1), -1)[1:(end - 1)]
end

"""
$TYPEDSIGNATURES

Encode the start airport of given activity as a one-hot vector.

`airports` should the list of all possible airports.
If the activity airport is not in the list, it is encoded as the "other" category.
"""
function onehot_start_airport(activity::AbstractActivity, airports::AbstractVector)
    return onehot(start_airport(activity), vcat(airports, "other"), "other")
end

"""
$TYPEDSIGNATURES

Alias for [`onehot_start_airport`](@ref).
"""
const onehot_departure_airport = onehot_start_airport

"""
$TYPEDSIGNATURES

Encode the end airport of given activity as a one-hot vector.

`airports` should the list of all possible airports.
If the activity airport is not in the list, it is encoded as the "other" category.
"""
function onehot_end_airport(activity::AbstractActivity, airports::AbstractVector)
    return onehot(end_airport(activity), vcat(airports, "other"), "other")
end

"""
$TYPEDSIGNATURES

Alias for [`onehot_end_airport`](@ref).
"""
const onehot_arrival_airport = onehot_end_airport

"""
$TYPEDSIGNATURES

Retrieve the departure airport size for given activity.
"""
function departure_airport_size(activity::AbstractActivity, airport_size_dict::Dict)
    return airport_size_dict[start_airport(activity)]
end

"""
$TYPEDSIGNATURES

Retrieve the arrival airport size for given activity.
"""
function arrival_airport_size(activity::AbstractActivity, airport_size_dict::Dict)
    return airport_size_dict[end_airport(activity)]
end

"""
$TYPEDEF

Abstract type defining a leg
"""
abstract type AbstractLeg <: AbstractActivity end

"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct Leg <: AbstractLeg
    "Unique identifier"
    id::String
    "IATA code of the departure airport"
    departure_airport::String
    "IATA code of the arrival airport"
    arrival_airport::String
    "departure time in UTC"
    departure_time::Dates.DateTime
    "arrival time in UTC"
    arrival_time::Dates.DateTime
    "aircraft type code"
    aircraft_type::String
    "minimum rurn time in minutes"
    minimum_turn_time::Int
end

"""
$TYPEDSIGNATURES

Check if an activity is a maintenance.
"""
is_maintenance(::Leg) = false

"""
$TYPEDSIGNATURES

Retrieve the start airport of given leg.
"""
start_airport(leg::Leg) = leg.departure_airport

"""
$TYPEDSIGNATURES

Retrieve the end airport of given leg.
"""
end_airport(leg::Leg) = leg.arrival_airport

"""
$TYPEDSIGNATURES

Retrieve the departure time of given leg.
"""
start_time(leg::Leg) = leg.departure_time

"""
$TYPEDSIGNATURES

Retrieve the arrival time of given leg.
"""
end_time(leg::Leg) = leg.arrival_time

"""
$TYPEDSIGNATURES

Retrieve the departure time of given leg.
"""
@inline departure_time(leg::Leg) = start_time(leg)

"""
$TYPEDSIGNATURES

Retrieve the arrival time of given leg.
"""
@inline arrival_time(leg::Leg) = end_time(leg)

"""
$TYPEDSIGNATURES

Retrieve the departure airport of given leg.
"""
@inline departure_airport(leg::Leg) = start_airport(leg)

"""
$TYPEDSIGNATURES

Retrieve the arrival airport of given leg.
"""
@inline arrival_airport(leg::Leg) = end_airport(leg)

"""
$TYPEDSIGNATURES

Retrieve the aircraft type of given leg.
"""
aircraft_type(leg::Leg) = leg.aircraft_type

"""
$TYPEDSIGNATURES

Encode the aircraft type of given leg as a one-hot vector.

`aircraft_types` should the list of all possible aircraft types.
If the activity type is not in the list, it is encoded as the "other" category.
"""
function onehot_aircraft_type(leg::AbstractLeg, aircraft_types::AbstractVector)
    return onehot(aircraft_type(leg), vcat(aircraft_types, "other"), "other")[1:(end - 1)]
end

"""
$TYPEDSIGNATURES

Retrieve the flight distance for given leg.
"""
function flight_distance(leg::Leg, distance_dict::Dict)
    key = start_airport(leg) * "-" * end_airport(leg)
    return distance_dict[key]
end

"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct Maintenance <: AbstractActivity
    "unique identifier"
    id::String
    "IATA code of the airport"
    location::String
    "start time of the maintenance in UTC"
    start_time::Dates.DateTime
    "end time of the maintenance in UTC"
    end_time::Dates.DateTime
    "immat code of the aircraft conserned by the maintenance"
    immat::String  # ? replace by an Immat
    "minimum turn time in minutes"
    minimum_turn_time::Int
end

"""
$TYPEDSIGNATURES

Check if an activity is a maintenance.
"""
is_maintenance(maintenance::Maintenance) = true

"""
$TYPEDSIGNATURES

Retrieve the location of given maintenance.
"""
start_airport(maintenance::Maintenance) = maintenance.location

"""
$TYPEDSIGNATURES

Retrieve the location of given maintenance.
"""
end_airport(maintenance::Maintenance) = maintenance.location

"""
$TYPEDSIGNATURES

Retrieve the start time of given maintenance.
"""
start_time(maintenance::Maintenance) = maintenance.start_time

"""
$TYPEDSIGNATURES

Retrieve the end time of given maintenance.
"""
end_time(maintenance::Maintenance) = maintenance.end_time

"""
$TYPEDSIGNATURES

Retrieve the minimum turn time of given maintenance.
"""
minimum_turn_time(maintenance::Maintenance) = 0.0  # ? Could add an option for choosing the type
# ? Could also add a towing in/towing out info if needed

"""
$TYPEDSIGNATURES

Retrieve the immat code associated to given maintenance.
"""
immat(maintenance::Maintenance) = maintenance.immat

"""
$TYPEDSIGNATURES

Retrieve the flight distance for given leg.
"""
function flight_distance(::Maintenance, ::Dict)
    return 0.0
end

"""
$TYPEDSIGNATURES

Check if two successive activities are location compatible, i.e. if the arrival airport of
`previous_activity` is the departure airport of `next_activity`.
"""
function is_location_compatible(
    previous_activity::AbstractActivity, next_activity::AbstractActivity
)
    return end_airport(previous_activity) == start_airport(next_activity)
end

"""
$TYPEDSIGNATURES

Return the slack time between two successive activities.

!!! warning
    This does not include the minimum turn time. For that, use [`slack_with_turn_time`](@ref).
"""
function slack(previous_activity::AbstractActivity, next_activity::AbstractActivity)
    return to_minutes(start_time(next_activity) - end_time(previous_activity))
end

function slack(m1::Maintenance, m2::Maintenance)
    return max(to_minutes(start_time(m2) - end_time(m1)), 0)
end

"""
$TYPEDSIGNATURES

Return the slack time between two successive activities, including the minimum turn time of the second activity.
slack(l₁, l₂) = (scheduled duration between arrival of l1 and departure of l2) - (minimum turn time)
"""
function slack_with_turn_time(
    previous_activity::AbstractActivity, next_activity::AbstractActivity
)
    return slack(previous_activity, next_activity) - minimum_turn_time(next_activity)
end

# ? is this correct?
function slack_with_turn_time(
    previous_activity::Maintenance, next_activity::AbstractActivity
)
    return slack(previous_activity, next_activity)
end

slack_with_turn_time(::Nothing, ::AbstractActivity) = 0.0
slack_with_turn_time(::AbstractActivity, ::Nothing) = 0.0
# AircraftRoutingBase.slack(::Nothing, ::Nothing) = 0.0

"""
$TYPEDSIGNATURES

Check if two successive activities are time compatible, i.e. if the slack between them is
larger than the minimum_turn_time of `next_activity`.
"""
function is_time_compatible(
    previous_activity::AbstractActivity, next_activity::AbstractActivity; TTM_factor=0.0
)
    return slack(previous_activity, next_activity) >=
           minimum_turn_time(next_activity) * TTM_factor
end

"""
$TYPEDSIGNATURES

Check if two successive maintenances are time compatible.
"""
function is_time_compatible(
    previous_maintenance::Maintenance, next_maintenance::Maintenance; TTM_factor=nothing
)
    # no self loop
    if previous_maintenance.id == next_maintenance.id
        return false
    end
    if end_time(previous_maintenance) <= start_time(next_maintenance)
        return true
    elseif end_time(next_maintenance) <= start_time(previous_maintenance)
        return false
    end
    # Else, the two maintenances overlap, there is a problem
    @error "Two maintenances are overlapping"
    @assert false
end

function is_aircraft_compatible(::AbstractActivity, ::AbstractActivity)
    return true
end

function is_aircraft_compatible(previous_leg::AbstractLeg, next_leg::AbstractLeg)
    return aircraft_type(previous_leg) == aircraft_type(next_leg)
end

function is_aircraft_compatible(previous_activity::Maintenance, next_activity::Maintenance)
    return immat(previous_activity) == immat(next_activity)
end

"""
$TYPEDSIGNATURES

Check if a given activity chaining (two successive activities) are compatible.
Additionaly prints a warning if the chaining is not compatible and `verbose` is true.
"""
function is_valid_chaining(
    previous_activity::AbstractActivity,
    next_activity::AbstractActivity;
    verbose=false,
    TTM_factor=0.0,
)
    if !is_aircraft_compatible(previous_activity, next_activity)
        verbose && @warn "Invalid chaining: aircrafts not compatible"
        return false
    end
    if !is_location_compatible(previous_activity, next_activity)
        verbose && @warn "Invalid chaining: locations not compatible"
        return false
    end
    if !is_time_compatible(previous_activity, next_activity; TTM_factor)
        verbose && @warn "Invalid chaining: times not compatible"
        return false
    end
    return true
end

"""
$TYPEDSIGNATURES

Return the chaining cost between two successive activities.
"""
function chaining_cost(::AbstractActivity, ::AbstractActivity; tractage_cost=0.0)
    return 0.0
end

"""
$TYPEDSIGNATURES

Return the chaining cost between two successive legs.
"""
function chaining_cost(
    previous_leg::AbstractLeg, next_leg::AbstractLeg; tractage_cost=1000.0
)
    Δ = slack(previous_leg, next_leg)
    if 4 * 60 <= Δ && Δ <= 6 * 60
        return tractage_cost
    end
    return 0.0
end
