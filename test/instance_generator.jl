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
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(; nb_legs=50, nb_aircraft=4, seed=42)

    root_delays = generate_root_delays(schedule; nb_scenarios=100, seed=42)

    @test size(root_delays) == (100, 50)
    @test eltype(root_delays) == Float32
    @test any(root_delays .< 0)
    @test -3 < mean(root_delays) < 3

    root_delays_2 = generate_root_delays(schedule; nb_scenarios=100, seed=42)
    @test root_delays == root_delays_2

    root_delays_3 = generate_root_delays(schedule; nb_scenarios=100, seed=99)
    @test root_delays_3 != root_delays

    # `delay_intensity` scales every root delay *after* shift and cap, so its
    # effect on the (capped) departure component is not a trivial 2x rescaling
    # of the population mean: check that the cap itself scales by 2 and is
    # actually reached, and that the mean still scales by about 2
    config = FeaturesConfig(; airports=schedule_airports(schedule))
    scenarios = DelayScenarios(schedule; nb_scenarios=100, seed=42, config)
    model1 = build_delay_model(schedule; config, delay_intensity=1.0)
    model2 = build_delay_model(schedule; config, delay_intensity=2.0)
    dep1 = sample_root_scenarios_unmerged(model1, scenarios).departure
    dep2 = sample_root_scenarios_unmerged(model2, scenarios).departure

    @test maximum(dep2) <= 2 * model1.max_dep
    @test maximum(dep2) ≈ 2 * model1.max_dep atol = 1.0
    @test mean(dep2) ≈ 2 * mean(dep1) rtol = 0.05
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

@testitem "Benchmark instance seat-weighted delay cost slopes" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    # a piecewise linear function's segment slopes are not exposed as a field, so
    # recompute them from consecutive breakpoints (plus the final `right_slope`)
    function segment_slopes(f)
        n = length(f.x)
        return [
            [(f.y[i + 1] - f.y[i]) / (f.x[i + 1] - f.x[i]) for i in 1:(n - 1)]
            f.right_slope
        ]
    end

    _, _, delay_cost_fn_unweighted = generate_benchmark_instance(50; nb_seats=1, seed=42)
    @test segment_slopes(delay_cost_fn_unweighted) ==
        FlightDelayModel.DELAY_COST_SLOPE_MEDIUM_HAUL

    _, _, delay_cost_fn_default = generate_benchmark_instance(50; seed=42)
    @test segment_slopes(delay_cost_fn_default) ==
        180 .* FlightDelayModel.DELAY_COST_SLOPE_MEDIUM_HAUL
end

@testitem "Benchmark instances are always coverable (regression)" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.AircraftRoutingBase

    for nb_legs_target in (50, 115), seed in (0, 1)
        schedule, _, _ = generate_benchmark_instance(nb_legs_target; nb_scenarios=2, seed)

        # the fleet built by the generator must always be able to cover every leg
        @test minimum_fleet_size(schedule) <= nb_immats(schedule)

        # a feasibility-only solve of the deterministic MIP must find a covering solution
        routes, _, _ = solve_aircraft_routing(
            schedule; silent=true, feasibility_only=true, time_limit=10.0
        )
        @test routes isa Vector{Route}
    end
end

@testitem "Minimum fleet size (hand-built schedule)" begin
    using Dates
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator

    # two legs departing at overlapping times from the same airport, but
    # arriving at different airports, cannot be chained in either order (an
    # aircraft cannot be in two places at once): 2 aircraft are needed
    overlapping_legs = [
        Leg(
            "1",
            "CDG",
            "JFK",
            DateTime(2022, 1, 1, 0, 0),
            DateTime(2022, 1, 1, 1, 0),
            "77W",
            0,
        ),
        Leg(
            "2",
            "CDG",
            "ORY",
            DateTime(2022, 1, 1, 0, 30),
            DateTime(2022, 1, 1, 1, 30),
            "77W",
            0,
        ),
    ]
    @test minimum_fleet_size(overlapping_legs) == 2

    # two legs that chain end-to-end (matching airport, enough turnaround
    # slack) only need a single aircraft
    chainable_legs = [
        Leg(
            "1",
            "JFK",
            "CDG",
            DateTime(2022, 1, 1, 0, 0),
            DateTime(2022, 1, 1, 4, 0),
            "77W",
            0,
        ),
        Leg(
            "2",
            "CDG",
            "ORY",
            DateTime(2022, 1, 1, 5, 0),
            DateTime(2022, 1, 1, 6, 0),
            "77W",
            0,
        ),
    ]
    @test minimum_fleet_size(chainable_legs) == 1
end
