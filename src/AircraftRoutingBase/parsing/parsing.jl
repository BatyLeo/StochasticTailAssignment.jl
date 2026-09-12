"""
$TYPEDSIGNATURES

Write given input schedule to custom json data format at `path`.
"""
function write_instance(instance::ActivitySchedule, path="instance.json")
    (; legs, maintenances, immats, forced_chainings) = instance

    df = Dates.DateFormat("yyyy-mm-ddTHH:MM:SSZ")

    leg_data = map(legs) do leg
        return Dict(
            "id" => leg.id,
            "departure_time" => Dates.format(start_time(leg), df),
            "arrival_time" => Dates.format(end_time(leg), df),
            "departure_airport" => start_airport(leg),
            "arrival_airport" => end_airport(leg),
            "aircraft_type" => aircraft_type(leg),
            "minimum_turn_time" => minimum_turn_time(leg),
        )
    end

    maintenance_data = map(maintenances) do maintenance
        return Dict(
            "id" => maintenance.id,
            "start_time" => Dates.format(start_time(maintenance), df),
            "end_time" => Dates.format(end_time(maintenance), df),
            "location" => maintenance.location,
            "immat" => maintenance.immat,
            "minimum_turn_time" => minimum_turn_time(maintenance),
        )
    end

    immat_data = map(immats) do immat
        return Dict(
            "immat" => immat.id,
            "fuel_factor" => immat.fuel_factor,
            "last_activity_id" => immat.last_activity_id,
            "aircraft_type" => immat.aircraft_type,
        )
    end

    forced_chaining_data = map(forced_chainings) do x
        return Dict(
            "first_activity_id" => x.first_activity_id,
            "second_activity_id" => x.second_activity_id,
        )
    end

    data = Dict(
        "activities" => Dict("legs" => leg_data, "maintenances" => maintenance_data),
        "immats" => immat_data,
        "forced_chainings" => forced_chaining_data,
    )

    open(path, "w") do f
        return JSON.print(f, data)
    end
end

"""
$TYPEDSIGNATURES

Read instance from custom json data format at `path`.
"""
function read_instance(
    path;
    standard_consumption::Dict{String,Float64},
    TTM_factor=0.0,
    include_chaining_costs=true,
    kwargs...,
)
    input = JSON.parsefile(path)
    leg_data = input["activities"]["legs"]
    maintenance_data = input["activities"]["maintenances"]
    immat_data = input["immats"]
    forced_chaining_data = input["forced_chainings"]

    leg_activities = map(leg_data) do x
        return Leg(;
            id=x["id"],
            departure_airport=x["departure_airport"],
            arrival_airport=x["arrival_airport"],
            departure_time=string_to_date(x["departure_time"]),
            arrival_time=string_to_date(x["arrival_time"]),
            aircraft_type=x["aircraft_type"],
            minimum_turn_time=x["minimum_turn_time"],
        )
    end
    maintenance_activities = map(maintenance_data) do x
        return Maintenance(;
            id=x["id"],
            location=x["location"],
            start_time=string_to_date(x["start_time"]),
            end_time=string_to_date(x["end_time"]),
            immat=x["immat"],
            minimum_turn_time=x["minimum_turn_time"],
        )
    end
    immats = map(immat_data) do x
        return Immat(
            x["immat"], x["fuel_factor"], x["last_activity_id"], x["aircraft_type"]
        )
    end

    forced_chainings = map(forced_chaining_data) do x
        return ForcedChaining(x["first_activity_id"], x["second_activity_id"])
    end

    return ActivitySchedule(;
        immats,
        legs=leg_activities,
        maintenances=maintenance_activities,
        forced_chainings=forced_chainings,
        standard_consumption,
        TTM_factor,
        include_chaining_costs,
        kwargs...,
    )
end
