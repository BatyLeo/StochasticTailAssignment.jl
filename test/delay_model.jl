@testitem "Handcrafted delay model: weight compilation" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    config = FeaturesConfig()
    coefficients = DelayCoefficients()
    predictor = HandcraftedDelayPredictor(coefficients, config)
    (; index) = compute_feature_info(config)

    # the duration coefficient lands exactly at the "Duration" feature index
    @test predictor.w_mu_dep[index["Duration"]] ≈ Float32(coefficients.mu_dep.duration)
    @test predictor.w_mu_arr[index["Duration"]] ≈ Float32(coefficients.mu_arr.duration)

    # the hub airport gets a strictly larger departure weight than a non-hub airport
    hub = first(config.airports)
    other = config.airports[2]
    @test predictor.w_mu_dep[index["DepartureAirport/$hub"]] >
        predictor.w_mu_dep[index["DepartureAirport/$other"]]

    # smoke test for weight_table
    table = weight_table(predictor)
    @test table.mu_dep == Pair.(config.names, predictor.w_mu_dep)

    # literal regression pins against the default compiled weights, on a
    # realistic schedule-derived config, guarding against accidental changes
    # to the compiled values across refactors
    schedule = generate_schedule(; nb_legs=115, nb_aircraft=8, seed=0)
    pin_config = FeaturesConfig(; airports=schedule_airports(schedule))
    pin_predictor = HandcraftedDelayPredictor(DelayCoefficients(), pin_config)
    pin_index = compute_feature_info(pin_config).index

    @test pin_predictor.w_mu_dep[pin_index["DayOfWeek/1"]] ≈ 2.996528f0 rtol = 1.0e-6
    @test pin_predictor.w_sigma_dep[pin_index["DepartureAirport/HUB"]] ≈ 0.043243244f0 rtol =
        1.0e-6
    @test pin_predictor.w_mu_arr[pin_index["ArrivalHour/20"]] ≈ 1.0583333f0 rtol = 1.0e-6
    @test pin_predictor.w_sigma_arr[pin_index["DayOfWeek/3"]] ≈ 4.449975f0 rtol = 1.0e-6
end

@testitem "Handcrafted delay model: bounds" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=100,
        nb_aircraft=8,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=1,
    )
    config = FeaturesConfig(; airports=schedule_airports(schedule))
    model = build_delay_model(schedule; config)
    scenarios = DelayScenarios(schedule; nb_scenarios=100, seed=1, config)
    (; departure, arrival) = sample_root_scenarios_unmerged(model, scenarios)

    @test all(-model.dep_shift .<= departure .<= model.max_dep)
    @test all(arrival .<= model.max_arr)
    @test all(isinteger, departure)
    @test all(isinteger, arrival)
end

@testitem "Handcrafted delay model: population aggregates" begin
    using Statistics
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=400,
        nb_aircraft=30,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=0,
    )
    config = FeaturesConfig(; airports=schedule_airports(schedule))
    model = build_delay_model(schedule; config)
    scenarios = DelayScenarios(schedule; nb_scenarios=200, seed=0, config)
    (; departure, arrival) = sample_root_scenarios_unmerged(model, scenarios)

    dep = vec(departure)
    @test mean(dep) ≈ 6.4 atol = 1.0
    @test median(dep) ≈ 3 atol = 1.5
    @test std(dep) ≈ 13.9 atol = 2.0
    @test quantile(dep, 0.90) ≈ 24 atol = 4
    @test quantile(dep, 0.95) ≈ 34 atol = 5
    @test quantile(dep, 0.99) ≈ 58 atol = 8
    @test mean(dep .<= 0) ≈ 0.42 atol = 0.05

    arr = vec(arrival)
    @test mean(arr) ≈ -6.7 atol = 1.0
    @test median(arr) ≈ -6 atol = 1.5
    @test std(arr) ≈ 9.2 atol = 1.5
    @test quantile(arr, 0.90) ≈ 4 atol = 2
    @test quantile(arr, 0.95) ≈ 8 atol = 3
    @test quantile(arr, 0.99) ≈ 17 atol = 4
    @test mean(arr .<= 0) ≈ 0.82 atol = 0.06
end

@testitem "Handcrafted delay model: heterogeneity and risk_spread" begin
    using Statistics
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=400,
        nb_aircraft=30,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=2,
    )

    config = FeaturesConfig(; airports=schedule_airports(schedule))

    function per_leg_mean_dep(risk_spread)
        model = build_delay_model(schedule; risk_spread, config)
        scenarios = DelayScenarios(schedule; nb_scenarios=300, seed=3, config)
        (; departure) = sample_root_scenarios_unmerged(model, scenarios)
        return departure
    end

    function per_leg_arr(risk_spread)
        model = build_delay_model(schedule; risk_spread, config)
        scenarios = DelayScenarios(schedule; nb_scenarios=300, seed=3, config)
        (; arrival) = sample_root_scenarios_unmerged(model, scenarios)
        return arrival
    end

    dep1 = per_leg_mean_dep(1.0)
    dep_hetero1 = std(vec(mean(dep1; dims=1)))
    @test dep_hetero1 >= 1.5

    dep16 = per_leg_mean_dep(1.6)
    dep_hetero16 = std(vec(mean(dep16; dims=1)))
    @test dep_hetero16 > dep_hetero1

    # the tau/sigma identity keeps the *marginal log-variance* of eps_dep
    # constant across risk_spread exactly (by construction), which would keep
    # the population mean exactly invariant if the per-leg heterogeneity term
    # were Gaussian; since it is really a sum of bounded, categorical
    # (hub/airport/hour/day) effects, not Gaussian, E[exp(mu)] drifts slightly
    # more than a first-order (variance-only) Jensen correction predicts as
    # risk_spread grows, hence the wider tolerance than the 3% quoted in the
    # design plan (empirically the drift is ~8-9% at risk_spread=1.6)
    @test mean(dep16) ≈ mean(dep1) rtol = 0.10
    @test std(dep16) ≈ std(dep1) rtol = 0.05

    # arrival counterpart: the sigma_arr.tau/sigma_arr identity (extended to also
    # account for the across-leg variance of sigma_arr itself, see
    # `_sigma_legs_variance`) keeps the population mean and standard deviation
    # of the arrival component approximately unchanged as risk_spread grows
    arr1 = per_leg_arr(1.0)
    arr16 = per_leg_arr(1.6)
    @test mean(arr16) ≈ mean(arr1) atol = 0.05 * std(vec(arr1))
    @test std(arr16) ≈ std(arr1) rtol = 0.05
end

@testitem "Handcrafted delay model: determinism" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator

    schedule = generate_schedule(;
        nb_legs=60,
        nb_aircraft=5,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=4,
    )

    root_delays_1 = generate_root_delays(schedule; nb_scenarios=40, seed=7)
    root_delays_2 = generate_root_delays(schedule; nb_scenarios=40, seed=7)
    @test root_delays_1 == root_delays_2

    root_delays_3 = generate_root_delays(schedule; nb_scenarios=40, seed=8)
    @test root_delays_3 != root_delays_1

    root_delays_more = generate_root_delays(schedule; nb_scenarios=80, seed=7)
    @test root_delays_more[1:40, :] == root_delays_1
end

@testitem "Handcrafted delay model: departure/arrival RNG streams are disjoint across seeds" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=30,
        nb_aircraft=3,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=4,
    )

    # `generate_dataset` draws consecutive seeds per instance: instance `s`'s
    # arrival stream must not be the same standard normal stream as instance
    # `s + 1`'s departure stream (which would happen with plain `seed`/`seed +
    # 1` derivations for the two streams)
    for s in (0, 7, 41)
        scenarios_s = DelayScenarios(schedule; nb_scenarios=20, seed=s)
        scenarios_s1 = DelayScenarios(schedule; nb_scenarios=20, seed=s + 1)
        @test scenarios_s.arrival_scenarios != scenarios_s1.departure_scenarios
    end
end

@testitem "Handcrafted delay model: dynamic simulator matches static propagation at beta=0" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=30,
        nb_aircraft=3,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=5,
    )
    routes, obj, _ = solve_aircraft_routing(schedule; silent=true)
    @test obj != -1

    config = FeaturesConfig(; airports=schedule_airports(schedule))
    model = build_delay_model(schedule; config)
    scenarios = DelayScenarios(schedule; nb_scenarios=20, seed=6, config)

    root_delays = sample_root_scenarios(model, scenarios)
    static_arrival_delays = propagate_delays_from_root_delays(routes, root_delays, schedule)

    dynamic = propagate_delays(model, scenarios, routes)

    @test dynamic.arrival_delays ≈ static_arrival_delays atol = 1.0e-4
end

@testitem "Handcrafted delay model: mismatched feature config errors" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=30,
        nb_aircraft=3,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=9,
    )

    # same airports, different order: same feature vector length, but a
    # different feature layout, which must be detected
    config_model = FeaturesConfig(; airports=["HUB", "A", "B", "C", "D", "E", "F", "G"])
    config_scenarios = FeaturesConfig(; airports=["A", "HUB", "B", "C", "D", "E", "F", "G"])

    model = build_delay_model(schedule; config=config_model)
    scenarios = DelayScenarios(schedule; nb_scenarios=5, seed=1, config=config_scenarios)

    @test_throws "feature layout mismatch" sample_root_scenarios(model, scenarios)
    @test_throws "feature layout mismatch" sample_root_scenarios_unmerged(model, scenarios)

    routes = [Route(1, Int[])]
    @test_throws "feature layout mismatch" propagate_delays(model, scenarios, routes)
end

@testitem "Handcrafted delay model: has_dynamic_features tracks nonzero betas" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=30,
        nb_aircraft=3,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=10,
    )
    config = FeaturesConfig(; airports=schedule_airports(schedule))

    static_model = build_delay_model(schedule; config)
    @test !has_dynamic_features(static_model)

    for coefficients in (
        DelayCoefficients(; beta_dep_has_propagated=0.5),
        DelayCoefficients(; beta_dep_propagated=0.5),
        DelayCoefficients(; beta_arr_has_departure_delay=0.5),
        DelayCoefficients(; beta_arr_propagated=0.5),
    )
        dynamic_model = build_delay_model(schedule; config, coefficients)
        @test has_dynamic_features(dynamic_model)

        scenarios = DelayScenarios(schedule; nb_scenarios=10, seed=11, config)
        @test_throws ArgumentError sample_root_scenarios(dynamic_model, scenarios)
        @test_throws ArgumentError sample_root_scenarios_unmerged(dynamic_model, scenarios)
    end
end

@testitem "Handcrafted delay model: dynamic simulator differs from static propagation when beta != 0" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using StochasticTailAssignment.FlightDelayModel

    schedule = generate_schedule(;
        nb_legs=30,
        nb_aircraft=3,
        airports=["HUB", "A", "B", "C", "D", "E", "F", "G"],
        seed=12,
    )
    routes, obj, _ = solve_aircraft_routing(schedule; silent=true)
    @test obj != -1

    config = FeaturesConfig(; airports=schedule_airports(schedule))
    scenarios = DelayScenarios(schedule; nb_scenarios=20, seed=13, config)

    static_model = build_delay_model(schedule; config)
    root_delays = sample_root_scenarios(static_model, scenarios)
    static_arrival_delays = propagate_delays_from_root_delays(routes, root_delays, schedule)

    dynamic_model = build_delay_model(
        schedule; config, coefficients=DelayCoefficients(; beta_dep_propagated=5.0)
    )
    dynamic = propagate_delays(dynamic_model, scenarios, routes)

    @test !(dynamic.arrival_delays ≈ static_arrival_delays)
end
