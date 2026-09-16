@testitem "Learning: generate_dataset labels and held-out scenarios" begin
    using StochasticTailAssignment

    nb_legs = 20
    data_integer = generate_dataset(2; nb_legs, nb_scenarios=20, seed=0, relaxation=false)
    data_relax = generate_dataset(2; nb_legs, nb_scenarios=20, seed=0, relaxation=true)

    for e in data_integer
        @test all(v -> v == 0 || v == 1, e.ȳ_integer)
        @test e.ȳ == e.ȳ_integer
    end

    # on these small instances the column generation relaxation is often integral
    # (frequent for this set-partitioning-like master), so fractionality is reported
    # but not required: the properties actually guaranteed are the value range and
    # the arc indexing shared with the integer label
    for (e_int, e_relax) in zip(data_integer, data_relax)
        ȳr = e_relax.ȳ_relaxation
        @test length(ȳr) == length(e_int.ȳ_integer)
        @test all(v -> -1e-6 <= v <= 1 + 1e-6, ȳr)
        @test e_relax.ȳ == ȳr
    end
    has_fractional_entry = any(
        any(v -> 1e-6 < v < 1 - 1e-6, e_relax.ȳ_relaxation) for e_relax in data_relax
    )
    has_fractional_entry ||
        @info "relaxation labels were integral on both test instances (allowed)"

    # held-out evaluation scenarios: same schedule, independent sampling from the
    # training scenarios used to build `x`, `routes` and the labels
    for e in data_integer
        @test size(e.eval_root_delays) == size(e.root_delays)
        @test e.eval_root_delays != e.root_delays
    end

    # the unmerged departure/arrival components stored alongside `root_delays` must sum
    # back to it exactly (same delay model, same delay seed, see `generate_dataset`)
    for e in data_integer
        @test e.departure_root_delays .+ e.arrival_root_delays ≈ e.root_delays
    end
end

@testitem "Learning: compute_features dimensions and slack quantiles" begin
    using StochasticTailAssignment
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.InstanceGenerator
    using Graphs: edges, src, dst
    using Statistics: quantile

    schedule = generate_schedule(; nb_legs=6, nb_aircraft=2, seed=0, store_arc_index=true)
    L = nb_legs(schedule)
    S = 5
    departure_root_delays = Float32.(reshape(1:(S * L), S, L))
    arrival_root_delays = Float32.(reshape((S * L):-1:1, S, L))

    x = compute_features(schedule, departure_root_delays, arrival_root_delays)
    @test size(x, 1) == 23
    @test size(x, 2) == schedule.nb_interior_arcs

    # direct recomputation of the slack quantiles for every leg-to-leg interior arc,
    # following the slack definition: the downstream leg `v`'s intrinsic delay only
    # affects its own departure, the upstream leg `u` contributes its arrival intrinsic
    # delay
    (; arc_index, immat_graphs) = schedule
    leg_to_leg_arcs = [
        (i, src(arc), dst(arc)) for i in 1:nb_immats(schedule) for
        arc in edges(immat_graphs[i]) if
        is_leg(schedule, src(arc)) && is_leg(schedule, dst(arc))
    ]
    @test !isempty(leg_to_leg_arcs)

    p = 0.1:0.1:1.0
    for (i, u, v) in leg_to_leg_arcs
        idx = arc_index[u, v, i]
        act_u = get_activity(schedule, u)
        act_v = get_activity(schedule, v)
        ω_tt = Float32(slack_with_turn_time(act_u, act_v))
        slack_without_v = ω_tt .- view(arrival_root_delays, :, u)
        slack_with_v = slack_without_v .+ view(departure_root_delays, :, v)

        @test x[3:12, idx] ≈ Float32.(quantile(slack_with_v, p))
        @test x[13:22, idx] ≈ Float32.(quantile(slack_without_v, p))
    end
end

@testitem "Learning: train_model! runs for both imitation targets" begin
    using StochasticTailAssignment

    nb_legs = 20
    data_train_raw = generate_dataset(2; nb_legs, nb_scenarios=20, seed=0)
    data_val_raw = generate_dataset(1; nb_legs, nb_scenarios=20, seed=100)

    dt = compute_normalization(data_train_raw)
    data_train = normalize_data(data_train_raw, dt)
    data_val = normalize_data(data_val_raw, dt)

    nb_features = size(data_train[1].x, 1)
    (; loss, maximizer) = build_loss(; ε=0.01, nb_samples=3)

    for (target, relaxation, field) in
        ((:integer, false, :ȳ_integer), (:relaxation, true, :ȳ_relaxation))
        model = default_model(nb_features; hidden_dims=[4])
        data_train_t = [(; e..., ȳ=getfield(e, field)) for e in data_train]
        data_val_t = [(; e..., ȳ=getfield(e, field)) for e in data_val]

        history = train_model!(
            model, loss, maximizer, data_train_t, data_val_t; nb_epochs=1, relaxation
        )

        @test length(history.loss_history) == 2
        @test length(history.metrics_val) == 2
        @test 0 <= history.best_epoch <= 1
        @test history.best_model isa typeof(model)

        m = evaluate_metrics(data_val, history.best_model, maximizer)
        @test isfinite(m.operational_cost_gap)
        @test isfinite(m.delay_cost_gap)
        @test isfinite(m.full_cost_gap)
        @test isfinite(m.avg_time)
    end
end
