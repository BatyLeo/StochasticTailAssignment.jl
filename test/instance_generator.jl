@testitem "Generate legs" begin
    using Dates
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator

    legs = generate_legs(;
        nb_legs=50,
        nb_aircraft=4,
        airports=["CDG", "ORY", "JFK", "LAX", "LHR", "FCO", "BCN", "AMS"],
        aircraft_type="320",
        horizon_start=DateTime(2025, 1, 6, 0, 0, 0),
        horizon_days=7,
        min_turn_time=25,
        seed=42,
    )

    @test length(legs) == 50
    @test all(l -> l.aircraft_type == "320", legs)
    @test all(l -> l.minimum_turn_time == 25, legs)
    @test all(l -> departure_airport(l) != arrival_airport(l), legs)
    @test all(l -> departure_time(l) < arrival_time(l), legs)
    @test all(l -> departure_time(l) >= DateTime(2025, 1, 6), legs)
    @test all(l -> arrival_time(l) <= DateTime(2025, 1, 13), legs)
    @test length(unique(l -> l.id, legs)) == 50
end

@testitem "Generate fleet" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator

    immats = generate_fleet(; nb_aircraft=5, aircraft_type="320", seed=42)

    @test length(immats) == 5
    @test all(im -> aircraft_type(im) == "320", immats)
    @test all(im -> im.last_activity_id == "s", immats)
    @test length(unique(im -> im.id, immats)) == 5
    @test all(im -> 1.0 <= im.fuel_factor <= 4.0, immats)
end

@testitem "Generate schedule" begin
    using Dates
    using Graphs
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator

    schedule = generate_schedule(;
        nb_legs=80,
        nb_aircraft=6,
        aircraft_type="320",
        horizon_start=DateTime(2025, 1, 6),
        horizon_days=7,
        seed=42,
    )

    @test nb_legs(schedule) == 80
    @test nb_immats(schedule) == 6
    @test nb_maintenances(schedule) == 0
    @test nv(schedule) > 80
    @test ne(schedule) > 80
end

@testitem "Generate root delays" begin
    using Dates
    using Statistics
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator

    schedule = generate_schedule(; nb_legs=50, nb_aircraft=4, seed=42)

    root_delays = generate_root_delays(schedule; nb_scenarios=100, seed=42)

    @test size(root_delays) == (100, 50)
    @test eltype(root_delays) == Float32
    @test all(root_delays .>= 0)
    @test mean(root_delays) > 0

    root_delays_2 = generate_root_delays(schedule; nb_scenarios=100, seed=42)
    @test root_delays == root_delays_2

    root_delays_3 = generate_root_delays(schedule; nb_scenarios=100, seed=99)
    @test root_delays_3 != root_delays
end

@testitem "End-to-end: generate, solve, evaluate" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(; nb_legs=50, nb_aircraft=4, horizon_days=7, seed=42)

    routes, obj, _ = solve_aircraft_routing(schedule)
    @test obj != -1
    @test is_feasible(routes, schedule)

    root_delays = generate_root_delays(schedule; nb_scenarios=50, seed=42)
    delay_cost_fn = DelayCostFunction(;
        slopes=FlightDelayModel.DELAY_COST_SLOPE_MEDIUM_HAUL
    )

    cost = full_cost(routes, root_delays, schedule; delay_cost_function=delay_cost_fn)
    @test isfinite(cost)

    op_cost = operational_cost(routes, schedule)
    delay_cost = delay_expected_cost(
        routes, root_delays, schedule; delay_cost_function=delay_cost_fn
    )
    @test cost ≈ op_cost + delay_cost
end

@testitem "Benchmark instance convenience" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.AircraftRoutingBase

    schedule, root_delays, delay_cost_fn = generate_benchmark_instance(
        50; nb_scenarios=30, seed=42
    )

    @test nb_legs(schedule) == 50
    @test size(root_delays, 1) == 30
    @test size(root_delays, 2) == 50

    schedule2, root_delays2, _ = generate_benchmark_instance(50; nb_scenarios=30, seed=42)
    @test nb_legs(schedule2) == 50
    @test root_delays == root_delays2

    schedule_big, root_delays_big, _ = generate_benchmark_instance(
        200; nb_scenarios=30, seed=42
    )
    @test nb_legs(schedule_big) == 200
end
