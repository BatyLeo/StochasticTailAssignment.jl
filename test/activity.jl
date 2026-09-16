@testitem "Legs" begin
    using Dates
    using StochasticTailAssignment.AircraftRoutingBase

    leg = Leg(;
        id="AAAA",
        departure_airport="CDG",
        arrival_airport="JFK",
        departure_time=string_to_date("2022-01-01T23:34:00"),
        arrival_time=string_to_date("2022-01-02T04:00:00"),
        aircraft_type="77W",
        minimum_turn_time=60,
    )

    # Constructor
    @test leg.id == "AAAA"
    @test leg.departure_airport == "CDG"
    @test leg.arrival_airport == "JFK"
    @test leg.departure_time == DateTime(2022, 1, 1, 23, 34, 00)
    @test leg.arrival_time == DateTime(2022, 1, 2, 4, 0, 0)
    @test leg.aircraft_type == "77W"
    @test leg.minimum_turn_time == 60

    # Accessors
    @test id(leg) == "AAAA"
    @test !is_maintenance(leg)
    @test start_airport(leg) == "CDG"
    @test departure_airport(leg) == start_airport(leg)
    @test end_airport(leg) == "JFK"
    @test arrival_airport(leg) == end_airport(leg)
    @test start_time(leg) == DateTime(2022, 1, 1, 23, 34, 00)
    @test departure_time(leg) == start_time(leg)
    @test end_time(leg) == DateTime(2022, 1, 2, 4, 0, 0)
    @test arrival_time(leg) == end_time(leg)
    @test aircraft_type(leg) == "77W"
    @test minimum_turn_time(leg) == 60

    @test duration(leg) == 266
    @test Dates.dayofweek(leg) == 6
    @test start_hour(leg) == 23
    @test departure_hour(leg) == start_hour(leg)
    encoding = falses(24)
    encoding[24] = true
    @test AircraftRoutingBase.onehot_start_hour(leg) == encoding
    @test AircraftRoutingBase.onehot_start_hour(leg, [1, 2, 3]) == [0, 0, 0]
    @test AircraftRoutingBase.onehot_start_hour(leg, [1, 23, 2]) == [0, 1, 0]
    @test end_hour(leg) == 4
    @test arrival_hour(leg) == end_hour(leg)
    encoding = falses(24)
    encoding[5] = true
    @test AircraftRoutingBase.onehot_end_hour(leg) == encoding
    @test AircraftRoutingBase.onehot_arrival_hour(leg) ==
        AircraftRoutingBase.onehot_end_hour(leg)
    @test AircraftRoutingBase.onehot_end_hour(leg, [1, 2, 23]) == [0, 0, 0]
    @test AircraftRoutingBase.onehot_end_hour(leg, [1, 4, 23]) == [0, 1, 0]
    @test Dates.month(leg) == 1
    @test AircraftRoutingBase.onehot_month(leg) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    @test AircraftRoutingBase.onehot_month(leg, [1, 3]) == [1, 0]
    @test AircraftRoutingBase.onehot_month(leg, 2:12) == falses(11)
    @test AircraftRoutingBase.onehot_departure_airport(leg, ["CDG", "JFK"]) == [1, 0, 0]
    @test AircraftRoutingBase.onehot_departure_airport(leg, ["ORY", "JFK"]) == [0, 0, 1]
    @test AircraftRoutingBase.onehot_arrival_airport(leg, ["CDG", "JFK"]) == [0, 1, 0]
    @test AircraftRoutingBase.onehot_arrival_airport(leg, ["CDG", "ORY"]) == [0, 0, 1]
    @test AircraftRoutingBase.onehot_aircraft_type(leg, ["77W", "333"]) == [1, 0]
end

@testitem "Maintenance" begin
    using Dates
    using StochasticTailAssignment.AircraftRoutingBase

    maintenance = Maintenance(;
        id="AAAA",
        location="CDG",
        start_time=DateTime(2022, 1, 1, 23, 34, 00),
        end_time=DateTime(2022, 1, 2, 4, 0, 0),
        immat="777",
        minimum_turn_time=60,
    )

    @test id(maintenance) == "AAAA"
    @test is_maintenance(maintenance)
    @test start_airport(maintenance) == "CDG"
    @test end_airport(maintenance) == start_airport(maintenance)
    @test start_time(maintenance) == DateTime(2022, 1, 1, 23, 34, 00)
    @test end_time(maintenance) == DateTime(2022, 1, 2, 4, 0, 0)
    @test minimum_turn_time(maintenance) == 0
    @test immat(maintenance) == "777"
    @test duration(maintenance) == 266
    @test Dates.dayofweek(maintenance) == 6
    @test start_hour(maintenance) == 23
    @test end_hour(maintenance) == 4
    @test Dates.month(maintenance) == 1
end

@testitem "Activity interaction" begin
    using StochasticTailAssignment.AircraftRoutingBase

    leg1 = Leg(;
        id="AAA1",
        departure_airport="CDG",
        arrival_airport="JFK",
        departure_time=string_to_date("2022-01-01T12:00:00"),
        arrival_time=string_to_date("2022-01-01T18:00:00"),
        aircraft_type="77W",
        minimum_turn_time=30,
    )
    leg2 = Leg(;
        id="AAA2",
        departure_airport="JFK",
        arrival_airport="LAX",
        departure_time=string_to_date("2022-01-01T20:00:00"),
        arrival_time=string_to_date("2022-01-02T02:00:00"),
        aircraft_type="77W",
        minimum_turn_time=60,
    )
    leg3 = Leg(;
        id="AAA2",
        departure_airport="JFK",
        arrival_airport="LAX",
        departure_time=string_to_date("2022-01-01T19:00:00"),
        arrival_time=string_to_date("2022-01-02T02:00:00"),
        aircraft_type="332",
        minimum_turn_time=120,
    )
    maintenance = Maintenance(;
        id="AAA3",
        location="CDG",
        start_time=string_to_date("2022-01-01T10:00:00"),
        end_time=string_to_date("2022-01-01T11:00:00"),
        immat="77W",
        minimum_turn_time=0,
    )
    @test AircraftRoutingBase.is_location_compatible(leg1, leg2)
    @test !AircraftRoutingBase.is_location_compatible(leg2, leg1)
    @test AircraftRoutingBase.is_time_compatible(leg1, leg2; TTM_factor=1.0)
    @test AircraftRoutingBase.is_aircraft_compatible(leg1, leg2)
    @test !AircraftRoutingBase.is_time_compatible(leg1, leg3; TTM_factor=1.0)
    @test !AircraftRoutingBase.is_aircraft_compatible(leg1, leg3)
    @test slack(leg1, leg2) == 120
    @test slack_with_turn_time(leg1, leg2) == 60
    @test is_valid_chaining(leg1, leg2; TTM_factor=1.0)
    @test !is_valid_chaining(leg1, leg3; TTM_factor=1.0)
    @test chaining_cost(leg1, leg2) == 0.0
    @test is_valid_chaining(maintenance, leg1; TTM_factor=1.0)
    @test chaining_cost(maintenance, leg1) == 0.0
end

@testitem "Schedule construction and feasibility" begin
    using StochasticTailAssignment.AircraftRoutingBase

    schedule = ActivitySchedule(;
        immats=[
            Immat("A", 1.0, "s", "77W"),
            Immat("B", 1.0, "s", "77W"),
            Immat("C", 1.0, "s", "77W"),
        ],
        standard_consumption=Dict("77W" => 125.0, "" => 65.0),
        legs=Leg[
            Leg(
                "1",
                "JFK",
                "CDG",
                string_to_date("2022-01-01T00:00"),
                string_to_date("2022-01-01T01:00"),
                "77W",
                0,
            )
            Leg(
                "2",
                "JFK",
                "CDG",
                string_to_date("2022-01-01T00:00"),
                string_to_date("2022-01-01T02:00"),
                "77W",
                0,
            )
            Leg(
                "3",
                "JFK",
                "CDG",
                string_to_date("2022-01-01T00:00"),
                string_to_date("2022-01-01T04:00"),
                "77W",
                0,
            )
            Leg(
                "4",
                "CDG",
                "JFK",
                string_to_date("2022-01-01T03:00"),
                string_to_date("2022-01-01T10:00"),
                "77W",
                0,
            )
            Leg(
                "5",
                "CDG",
                "JFK",
                string_to_date("2022-01-01T05:00"),
                string_to_date("2022-01-01T10:00"),
                "77W",
                0,
            )
            Leg(
                "6",
                "CDG",
                "JFK",
                string_to_date("2022-01-01T05:00"),
                string_to_date("2022-01-01T10:00"),
                "77W",
                0,
            )
        ],
        TTM_factor=1.0,
    )

    test_routes_1 = [Route(1, [1, 4]), Route(2, [2, 5]), Route(3, [3, 6])]
    test_routes_2 = [Route(1, [1, 5]), Route(2, [2, 4]), Route(3, [3, 6])]
    test_routes_3 = [Route(1, [1, 4]), Route(2, [2, 6]), Route(3, [3, 5])]
    test_routes_4 = [Route(1, [1, 5]), Route(2, [2, 6]), Route(3, [3, 4])]

    @test is_feasible(test_routes_1, schedule)
    @test is_feasible(test_routes_2, schedule)
    @test is_feasible(test_routes_3, schedule)
    @test !is_feasible(test_routes_4, schedule; verbose=false)
end

@testitem "Empty route feasibility" begin
    using StochasticTailAssignment.AircraftRoutingBase

    schedule = ActivitySchedule(;
        immats=[Immat("A", 1.0, "s", "77W"), Immat("B", 1.0, "s", "77W")],
        standard_consumption=Dict("77W" => 125.0, "" => 65.0),
        legs=Leg[Leg(
            "1",
            "JFK",
            "CDG",
            string_to_date("2022-01-01T00:00"),
            string_to_date("2022-01-01T01:00"),
            "77W",
            0,
        )],
        TTM_factor=1.0,
    )

    empty_route = Route(2, Int[])
    @test is_feasible(empty_route, schedule)

    routes = [Route(1, [1]), empty_route]
    @test is_feasible(routes, schedule)
end

@testitem "Empty route feasibility with forced last activity" begin
    using StochasticTailAssignment.AircraftRoutingBase

    # immat "A" is forced to start from leg "1" (its `last_activity_id`, a
    # real vertex of the graph, not the generic source "s"): an empty route
    # bypasses that forced chaining and must be reported as infeasible
    schedule = ActivitySchedule(;
        immats=[Immat("A", 1.0, "1", "77W")],
        standard_consumption=Dict("77W" => 125.0, "" => 65.0),
        legs=Leg[Leg(
            "1",
            "JFK",
            "CDG",
            string_to_date("2022-01-01T00:00"),
            string_to_date("2022-01-01T01:00"),
            "77W",
            0,
        )],
        TTM_factor=1.0,
    )

    empty_route = Route(1, Int[])
    @test !is_feasible(empty_route, schedule; verbose=false)
end
