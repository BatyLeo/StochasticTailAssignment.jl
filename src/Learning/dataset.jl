"""
$TYPEDSIGNATURES

Generate a learning dataset of `nb_instances` labeled examples for the stochastic tail
assignment pricing problem.

Each instance is generated with [`generate_benchmark_instance`](@ref) (with
`store_arc_index=true`), solved to (near-)optimality with the deterministic MIP followed
by the stochastic column generation and diving heuristic, and labeled with the resulting
expert routes.

If `relaxation` is `true`, the expert arc labels `ȳ` are the LP relaxation arc marginals
returned by the column generation (for training with `relaxation=true` in
[`train_model!`](@ref)). If `relaxation` is `false` (the default), `ȳ` is the binary arc
solution decoded from the diving heuristic routes. In both cases, `routes` is always
computed via the diving heuristic (needed for evaluation/baseline).

Both expert labels are always computed and returned regardless of `relaxation`, so that a
single dataset can be reused to train against either target: `ȳ_integer` is the binary arc
solution decoded from the diving heuristic routes (target `D^Y`), and `ȳ_relaxation` is the
fractional arc-incidence vector of the column generation relaxation, a convex combination of
columns with entries in `[0, 1]` (target `D^Ỹ`). Both share the same arc indexing as `ȳ` and
as the edge MIP maximizer (see [`aircraft_routing_edge_maximizer`](@ref)).

`root_delays` is the scenario matrix used to build the features `x` and to solve for the
expert labels (`routes`, `ȳ_integer`, `ȳ_relaxation`). `eval_root_delays` is a second,
held-out `nb_eval_scenarios x nb_legs` root delay matrix for the same schedule, sampled
independently (with a seed offset by `10_000` from `root_delays`'s), meant to be used for
evaluation (see [`evaluate_metrics`](@ref)) instead of the training scenarios.

`departure_root_delays` and `arrival_root_delays` are the unmerged departure and arrival
components of `root_delays` (see [`sample_root_scenarios_unmerged`](@ref)), sampled from
the same [`StochasticTailAssignment.FlightDelayModel.SyntheticDelayModel`](@ref) and the same delay seed as `root_delays` (so that
`departure_root_delays .+ arrival_root_delays == root_delays`), and passed to
[`compute_features`](@ref) to build `x`. If `nb_feature_scenarios` differs from
`nb_scenarios`, `departure_root_delays` and `arrival_root_delays` are instead drawn from a
separate, unmerged scenario set of size `nb_feature_scenarios` (same delay model, seed
offset by `20_000` from the delay seed), used only to compute `x`, not for `root_delays` or
the expert labels. A larger `nb_feature_scenarios` gives smoother slack quantile features,
at the cost of extra sampling.

Returns a `Vector` of `NamedTuple`s with fields `instance`, `x` (features), `ȳ` (expert
arc solution selected by `relaxation`), `ȳ_integer`, `ȳ_relaxation`, `routes` (expert
routes, from the diving heuristic), `root_delays`, `departure_root_delays`,
`arrival_root_delays`, `eval_root_delays`, `delay_cost_function`, `expert_time` (wall-clock
time of the stochastic column generation plus diving heuristic call only, excluding
instance generation and the deterministic MIP warm start, suitable for a fair "expert
solving time" comparison).

`risk_spread`, `delay_intensity`, `nb_seats` and `nb_extra_aircraft` are forwarded to
[`generate_benchmark_instance`](@ref).

`max_discrepancy` is the taboo list size limit forwarded to
[`diving_heuristic_with_backtracking!`](@ref).
"""
function generate_dataset(
    nb_instances::Int;
    nb_legs=20,
    nb_scenarios=50,
    nb_eval_scenarios=nb_scenarios,
    nb_feature_scenarios::Int=nb_scenarios,
    seed=0,
    silent=true,
    relaxation=false,
    model_builder=highs_model,
    max_nb_columns=10000,
    tol=1e-6,
    time_limit::Union{Nothing,Float64}=nothing,
    risk_spread=1.0,
    delay_intensity=1.0,
    nb_seats=180,
    nb_extra_aircraft::Int=0,
    max_discrepancy::Int=2,
    kwargs...,
)
    return map(1:nb_instances) do i
        delay_seed = seed + i + 100
        schedule, root_delays, delay_cost_function = generate_benchmark_instance(
            nb_legs;
            nb_scenarios,
            seed=seed + i,
            delay_seed,
            store_arc_index=true,
            risk_spread,
            delay_intensity,
            nb_seats,
            nb_extra_aircraft,
        )

        # unmerged departure/arrival components of `root_delays`, sampled from the same
        # delay model and the same delay seed (see `generate_benchmark_instance`'s
        # `delay_seed` keyword), so that `departure_root_delays .+ arrival_root_delays ==
        # root_delays`
        config = FeaturesConfig(; airports=schedule_airports(schedule))
        delay_model = build_delay_model(schedule; delay_intensity, risk_spread, config)
        delay_scenarios = DelayScenarios(schedule; nb_scenarios, config, seed=delay_seed)
        unmerged_root_delays = sample_root_scenarios_unmerged(delay_model, delay_scenarios)

        # `nb_feature_scenarios == nb_scenarios` (the default) reuses `unmerged_root_delays`
        # so that the default behavior is unchanged. Otherwise, a separate scenario set is
        # drawn (same delay model, seed offset by `20_000`), used only to compute `x`.
        if nb_feature_scenarios == nb_scenarios
            departure_root_delays = unmerged_root_delays.departure
            arrival_root_delays = unmerged_root_delays.arrival
        else
            feature_delay_scenarios = DelayScenarios(
                schedule;
                nb_scenarios=nb_feature_scenarios,
                config,
                seed=delay_seed + 20_000,
            )
            feature_unmerged_root_delays = sample_root_scenarios_unmerged(
                delay_model, feature_delay_scenarios
            )
            departure_root_delays = feature_unmerged_root_delays.departure
            arrival_root_delays = feature_unmerged_root_delays.arrival
        end

        eval_root_delays = generate_root_delays(
            schedule;
            nb_scenarios=nb_eval_scenarios,
            seed=seed + i + 10_000,
            risk_spread,
            delay_intensity,
        )

        det_routes, _, _ = solve_aircraft_routing(schedule; silent, time_limit)
        det_routes isa Vector{Route} ||
            error("deterministic MIP is infeasible for the generated instance")

        expert_time = @elapsed begin
            cg = stochastic_column_generation(
                schedule,
                det_routes;
                model_builder,
                root_delays,
                delay_cost_function,
                max_nb_columns,
                tol,
                silent,
                kwargs...,
            )

            routes, feasible = diving_heuristic_with_backtracking!(
                schedule,
                cg.columns,
                root_delays,
                cg.dual_values,
                max_discrepancy;
                model_builder,
                delay_cost_function,
                silent,
            )
        end
        feasible || error(
            "diving heuristic found no feasible solution (max_discrepancy=$max_discrepancy)",
        )
        # `diving_heuristic_with_backtracking!` is typed `Union{Route,Array}` (it is a
        # ported function, kept as-is), so narrow it here to a concrete `Vector{Route}`
        # once `feasible` has been confirmed, instead of leaking the wider union type
        routes isa AbstractVector{<:Route} || throw(
            ArgumentError(
                "diving_heuristic_with_backtracking! returned a $(typeof(routes)), " *
                "expected a Vector{Route}",
            ),
        )
        routes = Vector{Route}(routes)

        x = compute_features(schedule, departure_root_delays, arrival_root_delays)
        ȳ_integer = decode_arc_solution_from_routes(routes, schedule)
        ȳ_relaxation = cg.y_val_relax
        ȳ = relaxation ? ȳ_relaxation : ȳ_integer
        return (;
            instance=schedule,
            x,
            ȳ,
            ȳ_integer,
            ȳ_relaxation,
            routes,
            root_delays,
            departure_root_delays,
            arrival_root_delays,
            eval_root_delays,
            delay_cost_function,
            expert_time,
        )
    end
end

"""
$TYPEDSIGNATURES

Compute per-feature normalization statistics (`mins`, `ranges`) over the feature matrices
of `data`, as produced by [`generate_dataset`](@ref).
"""
function compute_normalization(data)
    X = hcat([e.x for e in data]...)
    mins = minimum(X; dims=2)
    maxs = maximum(X; dims=2)
    ranges = maxs .- mins
    ranges[ranges .== 0] .= 1
    return (; mins, ranges)
end

"""
$TYPEDSIGNATURES

Normalize the feature matrices of `data` using the normalization statistics `dt` computed
by [`compute_normalization`](@ref).
"""
function normalize_data(data, dt)
    return [(; e..., x=Float32.((e.x .- dt.mins) ./ dt.ranges)) for e in data]
end
