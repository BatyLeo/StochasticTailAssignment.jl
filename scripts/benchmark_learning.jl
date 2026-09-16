"""
    benchmark_learning.jl

Benchmark the decision-focused learning (DFL) imitation pipeline on generated instances:
a feedforward model is trained by imitation of expert labels, using a Fenchel-Young loss
around the edge-based MIP pricing oracle.

Two imitation targets are compared: `D^Y`, the binary arc solution decoded from the diving
heuristic's integer routes, and `D^Ỹ`, the fractional arc vector of the column generation
relaxation (a convex combination of columns). In both cases, the expert used to compute
the gap is the same integer diving solution, and all gaps are measured on a held-out
scenario set (`eval_root_delays`, see `generate_dataset`), distinct from the scenarios
used to build features and expert labels.

Run from the package root:

    julia --project=scripts scripts/benchmark_learning.jl

Pass `--small` to run a quick smoke test with tiny instances and few epochs:

    julia --project=scripts scripts/benchmark_learning.jl --small

For each instance size in `NB_LEGS_LIST`, the script:
1. generates train/validation/test datasets once with `generate_dataset`, which stores
   both expert labels (`ȳ_integer` and `ȳ_relaxation`) and the held-out evaluation
   scenarios, so the two models below see identical instances,
2. computes baselines on the test set (deterministic MIP, expert diving heuristic),
   evaluated on the held-out scenarios,
3. trains, for each target in `TARGETS`, a small MLP by imitation with `train_model!`,
   starting from the same seed and hyperparameters, and selects the best validation-gap
   epoch as the model to evaluate (see `train_model!`),
4. evaluates the selected model of each target on the test set,
5. prints a per-method cost summary table on the test set (deterministic MIP, both
   imitation models, expert diving heuristic), and a comparison table of the gap to the
   expert across train/validation/test splits for both imitation targets and the
   deterministic MIP.
"""

using StochasticTailAssignment
using StochasticTailAssignment.AircraftRoutingBase
using StochasticTailAssignment.InstanceGenerator
using StochasticTailAssignment.FlightDelayModel
using Printf
using Random
using Statistics: mean

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

small_mode = "--small" in ARGS

const NB_LEGS_LIST = small_mode ? [8] : [60, 100]  # instance sizes to benchmark
const NB_SCENARIOS = small_mode ? 20 : 100  # training scenarios per instance
const NB_EVAL_SCENARIOS = small_mode ? 20 : 100  # held-out scenarios per instance
const NB_TRAIN = small_mode ? 3 : 40  # number of training instances
const NB_VAL = small_mode ? 1 : 10  # number of validation instances
const NB_TEST = small_mode ? 2 : 10  # number of test instances
const NB_EPOCHS = small_mode ? 3 : 100  # training epochs
const ε = 0.01  # perturbation scale of the Fenchel-Young loss
const NB_SAMPLES = small_mode ? 5 : 10  # Monte-Carlo samples of the perturbed loss
const HIDDEN_DIMS = [10, 10]  # hidden layer sizes of the imitation model
const TARGETS = (:integer, :relaxation)  # imitation targets compared by the benchmark
const RISK_SPREAD = 1.0  # instance generation parameter, forwarded to generate_dataset
const DELAY_INTENSITY = 1.0  # instance generation parameter, forwarded to generate_dataset
const NB_SEATS = 180  # instance generation parameter, forwarded to generate_dataset
const NB_EXTRA_AIRCRAFT = 0  # instance generation parameter, forwarded to generate_dataset

const TRAIN_SEED = 0  # base seed for the training dataset
const VAL_SEED = 1_000_000  # base seed for the validation dataset
const TEST_SEED = 2_000_000  # base seed for the test dataset
const MODEL_SEED = 1234  # seed for model weight initialization
const MAX_ATTEMPTS = 20  # must match `generate_dataset_safe`'s `max_attempts` default

const TARGET_LABEL = Dict(  # display name of each imitation target
    :integer => "Imitation D^Y (integer)",
    :relaxation => "Imitation D^Ỹ (relaxation)",
)

if small_mode
    @info "Running in --small mode (smoke test parameters)"
end

# ---------------------------------------------------------------------------
# Seed range disjointness check
# ---------------------------------------------------------------------------

"""
Range of seeds that `generate_dataset_safe(n, base_seed; max_attempts, ...)` can draw
from, following its `seed = base_seed + (i - 1) * 1000 + attempt` derivation (`i` ranges
over `1:n`, `attempt` over `0:max_attempts`).
"""
function seed_range(base_seed, n; max_attempts=MAX_ATTEMPTS)
    return base_seed:(base_seed + (n - 1) * 1000 + max_attempts)
end

# the train/val/test base seeds must be spaced widely enough that their
# `generate_dataset_safe` seed ranges never overlap (a past `+ 10_000`/`+ 20_000` spacing
# let training instance 11 (and 21) collide with validation (and test) instance 1)
let
    splits = [
        ("train", seed_range(TRAIN_SEED, NB_TRAIN)),
        ("val", seed_range(VAL_SEED, NB_VAL)),
        ("test", seed_range(TEST_SEED, NB_TEST)),
    ]
    for i in eachindex(splits), j in (i + 1):length(splits)
        name_i, range_i = splits[i]
        name_j, range_j = splits[j]
        isempty(intersect(range_i, range_j)) || error(
            "seed ranges for the $name_i and $name_j splits overlap: $range_i vs $range_j",
        )
    end
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

print_header(title) = (println("=" ^ 80); println(title); println("=" ^ 80))

"""
Generate `nb_instances` dataset entries (see `generate_dataset`), one at a time, starting
from seeds derived from `base_seed`.

`generate_dataset`'s diving heuristic can exhaust its `max_discrepancy` taboo list before
finding a feasible integer solution on some instance/seed combinations, raising an error.
When this happens, retry with a perturbed seed instead of failing the whole benchmark.

Every dataset entry stores both expert labels (`ȳ_integer` and `ȳ_relaxation`, see
`generate_dataset`), so it can be reused to train against either imitation target, and a
held-out evaluation scenario set (`eval_root_delays`).

`risk_spread`, `delay_intensity`, `nb_seats` and `nb_extra_aircraft` are forwarded to
`generate_dataset` (and, from there, to `generate_benchmark_instance`).

Returns the vector of dataset entries and the wall-clock time taken to generate each one.
"""
function generate_dataset_safe(
    nb_instances,
    base_seed;
    nb_legs,
    nb_scenarios,
    nb_eval_scenarios=nb_scenarios,
    max_attempts=20,
    risk_spread=1.0,
    delay_intensity=1.0,
    nb_seats=180,
    nb_extra_aircraft=0,
)
    data = []
    times = Float64[]
    for i in 1:nb_instances
        attempt = 0
        while true
            seed = base_seed + (i - 1) * 1000 + attempt
            try
                t = @elapsed d = generate_dataset(
                    1;
                    nb_legs,
                    nb_scenarios,
                    nb_eval_scenarios,
                    seed,
                    risk_spread,
                    delay_intensity,
                    nb_seats,
                    nb_extra_aircraft,
                )[1]
                push!(data, d)
                push!(times, t)
                break
            catch err
                attempt += 1
                attempt > max_attempts && rethrow()
                @warn "generate_dataset failed (instance $i, seed=$seed), retrying with a perturbed seed" exception =
                    err
            end
        end
    end
    return data, times
end

"""
Return a copy of dataset `data` with the `ȳ` field replaced by the label matching
imitation `target` (`:integer` or `:relaxation`), so it can be passed to `train_model!`.
"""
function select_target(data, target::Symbol)
    field = target === :integer ? :ȳ_integer : :ȳ_relaxation
    return [(; e..., ȳ=getfield(e, field)) for e in data]
end

"""
Solve each instance in `data` with the deterministic MIP (no delay information) and
evaluate the resulting routes against the true stochastic cost (on `eval_root_delays`,
the held-out scenario set), as a "value of stochastic information" baseline.
"""
function compute_deterministic_baseline(data; time_limit::Union{Nothing,Float64}=nothing)
    results = NamedTuple[]
    for e in data
        (; instance, eval_root_delays, delay_cost_function) = e
        t = @elapsed begin
            det_routes, _, _ = solve_aircraft_routing(instance; silent=true, time_limit)
        end
        det_routes isa Vector{Route} ||
            error("deterministic MIP is infeasible on a dataset instance")
        op = operational_cost(det_routes, instance)
        full = full_cost(det_routes, eval_root_delays, instance; delay_cost_function)
        push!(results, (; op_cost=op, delay_cost=full - op, full_cost=full, time=t))
    end
    return results
end

"""
Read off the expert (diving heuristic) cost breakdown from dataset entries produced by
`generate_dataset`, evaluated on `eval_root_delays`, paired with `expert_time` (the
wall-clock time of the column generation plus diving heuristic call only, see
`generate_dataset`), so the "Time (s)" column of the summary table reflects just the
expert solve, not instance generation or the deterministic MIP warm start.
"""
function expert_baseline(data)
    results = NamedTuple[]
    for e in data
        (; routes, instance, eval_root_delays, delay_cost_function, expert_time) = e
        op = operational_cost(routes, instance)
        full = full_cost(routes, eval_root_delays, instance; delay_cost_function)
        push!(
            results, (; op_cost=op, delay_cost=full - op, full_cost=full, time=expert_time)
        )
    end
    return results
end

"""
Read off the expert (diving heuristic) cost breakdown from dataset entries, evaluated on
`eval_root_delays`, without a timing wrapper (used for the train/validation splits, which
are not individually timed).
"""
function expert_costs(data)
    results = NamedTuple[]
    for e in data
        (; routes, instance, eval_root_delays, delay_cost_function) = e
        op = operational_cost(routes, instance)
        full = full_cost(routes, eval_root_delays, instance; delay_cost_function)
        push!(results, (; op_cost=op, delay_cost=full - op, full_cost=full))
    end
    return results
end

"""
Run the trained `model` through the `maximizer` (edge MIP pricing oracle) on each test
instance and evaluate the resulting routes, on `eval_root_delays`.
"""
function imitation_baseline(data, model, maximizer)
    results = NamedTuple[]
    for e in data
        (; instance, x, eval_root_delays, delay_cost_function) = e
        res = @timed begin
            θ = model(x)
            maximizer(θ; instance)
        end
        routes = decode_routes_from_arc_solution(res.value, instance)
        op = operational_cost(routes, instance)
        full = full_cost(routes, eval_root_delays, instance; delay_cost_function)
        push!(results, (; op_cost=op, delay_cost=full - op, full_cost=full, time=res.time))
    end
    return results
end

"""
Aggregate per-instance results into average costs, average relative gap to the expert
(in %), and average solving time.
"""
function summarize(results, expert_results)
    avg_op = mean(r.op_cost for r in results)
    avg_delay = mean(r.delay_cost for r in results)
    avg_full = mean(r.full_cost for r in results)
    avg_time = mean(r.time for r in results)
    gaps = [
        100 * (r.full_cost - e.full_cost) / abs(e.full_cost) for
        (r, e) in zip(results, expert_results) if e.full_cost != 0
    ]
    avg_gap = isempty(gaps) ? NaN : mean(gaps)
    return (; avg_op, avg_delay, avg_full, avg_gap, avg_time)
end

"""
Average relative gap to the expert of operational, delay and full cost, over paired
`results` and `expert_results` (each a vector of `NamedTuple`s with fields `op_cost`,
`delay_cost`, `full_cost`), following the same convention as `evaluate_metrics`: each gap
is averaged only over the instances where the corresponding expert cost is strictly
positive, not over `length(results)`.
"""
function compute_gaps(results, expert_results)
    total_op = 0.0
    total_delay = 0.0
    total_full = 0.0
    nb_op = 0
    nb_delay = 0
    nb_full = 0
    for (r, e) in zip(results, expert_results)
        if e.op_cost > 0
            total_op += (r.op_cost - e.op_cost) / abs(e.op_cost)
            nb_op += 1
        end
        if e.delay_cost > 0
            total_delay += (r.delay_cost - e.delay_cost) / abs(e.delay_cost)
            nb_delay += 1
        end
        if e.full_cost > 0
            total_full += (r.full_cost - e.full_cost) / abs(e.full_cost)
            nb_full += 1
        end
    end
    return (;
        op_gap=nb_op > 0 ? total_op / nb_op : NaN,
        delay_gap=nb_delay > 0 ? total_delay / nb_delay : NaN,
        full_gap=nb_full > 0 ? total_full / nb_full : NaN,
    )
end

"""
Print a table under header line `title`: pre-formatted `header` row, a separator of
`width` dashes, one pre-formatted line per element of `rows`, and a trailing blank line.
Shared by [`print_summary_table`](@ref) and [`print_comparison_table`](@ref).
"""
function print_table(title, width, header, rows)
    print_header(title)
    println(header)
    println("-" ^ width)
    for row in rows
        println(row)
    end
    return println()
end

function print_summary_table(n_legs, methods, expert_results)
    header = @sprintf(
        "%-30s  %12s  %12s  %12s  %10s  %10s",
        "Method",
        "Op cost",
        "Delay cost",
        "Full cost",
        "Gap to exp",
        "Time (s)"
    )
    rows = map(methods) do (name, results)
        s = summarize(results, expert_results)
        gap_str =
            name == "Expert (diving heuristic)" ? "-" : @sprintf("%+9.1f%%", s.avg_gap)
        @sprintf(
            "%-30s  %12.1f  %12.1f  %12.1f  %10s  %10.4f",
            name,
            s.avg_op,
            s.avg_delay,
            s.avg_full,
            gap_str,
            s.avg_time,
        )
    end
    return print_table(
        "Summary for nb_legs = $n_legs (test set, n=$(length(expert_results)))",
        94,
        header,
        rows,
    )
end

"""
Print the comparison table between imitation targets: relative gap to the
expert (full cost) on train/validation/test, plus the operational and delay cost gap on
test, for each row in `rows` (a vector of `(name, gaps_by_split)` pairs, where
`gaps_by_split` is a `NamedTuple` with fields `train`, `val`, `test`, each themselves a
`NamedTuple` with fields `op_gap`, `delay_gap`, `full_gap`, as returned by
`compute_gaps`/`evaluate_metrics`).
"""
function print_comparison_table(n_legs, rows)
    header = @sprintf(
        "%-30s  %10s  %10s  %10s  %14s  %14s",
        "Method",
        "Train gap",
        "Val gap",
        "Test gap",
        "Test op gap",
        "Test delay gap"
    )
    formatted_rows = map(rows) do (name, g)
        @sprintf(
            "%-30s  %+9.1f%%  %+9.1f%%  %+9.1f%%  %+13.1f%%  %+13.1f%%",
            name,
            100 * g.train.full_gap,
            100 * g.val.full_gap,
            100 * g.test.full_gap,
            100 * g.test.op_gap,
            100 * g.test.delay_gap,
        )
    end
    return print_table(
        "Imitation targets comparison for nb_legs = $n_legs (D^Y vs D^Ỹ)",
        100,
        header,
        formatted_rows,
    )
end

function print_loss_curve(target, history)
    print_header("Training loss curve ($(TARGET_LABEL[target]))")
    @printf(
        "%-6s  %12s  %12s  %14s  %13s\n",
        "Epoch",
        "Loss",
        "Val op gap",
        "Val delay gap",
        "Val full gap"
    )
    println("-" ^ 65)
    nb_points = length(history.loss_history)
    for epoch in 0:(nb_points - 1)
        m = history.metrics_val[epoch + 1]
        marker = epoch == history.best_epoch ? "  <- best" : ""
        @printf(
            "%-6d  %12.4f  %11.1f%%  %13.1f%%  %12.1f%%%s\n",
            epoch,
            history.loss_history[epoch + 1],
            100 * m.operational_cost_gap,
            100 * m.delay_cost_gap,
            100 * m.full_cost_gap,
            marker,
        )
    end
    return println()
end

# ---------------------------------------------------------------------------
# Main benchmark for a given instance size
# ---------------------------------------------------------------------------

function run_benchmark_for_size(n_legs)
    print_header("Benchmark for nb_legs = $n_legs")

    @info "Generating training dataset ($NB_TRAIN instances)..."
    t_train = @elapsed (data_train_raw, _) = generate_dataset_safe(
        NB_TRAIN,
        TRAIN_SEED;
        nb_legs=n_legs,
        nb_scenarios=NB_SCENARIOS,
        nb_eval_scenarios=NB_EVAL_SCENARIOS,
        risk_spread=RISK_SPREAD,
        delay_intensity=DELAY_INTENSITY,
        nb_seats=NB_SEATS,
        nb_extra_aircraft=NB_EXTRA_AIRCRAFT,
    )
    @info "  done in $(round(t_train; digits=1))s"

    @info "Generating validation dataset ($NB_VAL instances)..."
    t_val = @elapsed (data_val_raw, _) = generate_dataset_safe(
        NB_VAL,
        VAL_SEED;
        nb_legs=n_legs,
        nb_scenarios=NB_SCENARIOS,
        nb_eval_scenarios=NB_EVAL_SCENARIOS,
        risk_spread=RISK_SPREAD,
        delay_intensity=DELAY_INTENSITY,
        nb_seats=NB_SEATS,
        nb_extra_aircraft=NB_EXTRA_AIRCRAFT,
    )
    @info "  done in $(round(t_val; digits=1))s"

    @info "Generating test dataset ($NB_TEST instances, timed individually)..."
    data_test_raw, gen_times = generate_dataset_safe(
        NB_TEST,
        TEST_SEED;
        nb_legs=n_legs,
        nb_scenarios=NB_SCENARIOS,
        nb_eval_scenarios=NB_EVAL_SCENARIOS,
        risk_spread=RISK_SPREAD,
        delay_intensity=DELAY_INTENSITY,
        nb_seats=NB_SEATS,
        nb_extra_aircraft=NB_EXTRA_AIRCRAFT,
    )
    @info "  done in $(round(sum(gen_times); digits=1))s total"

    nb_features = size(data_train_raw[1].x, 1)
    @info "Dataset statistics: nb_features=$nb_features, nb_train=$(length(data_train_raw)), nb_val=$(length(data_val_raw)), nb_test=$(length(data_test_raw))"
    for (label, d) in
        (("train", data_train_raw), ("val", data_val_raw), ("test", data_test_raw))
        arcs = [size(e.x, 2) for e in d]
        legs = [nb_legs(e.instance) for e in d]
        immats = [nb_immats(e.instance) for e in d]
        @info "  $label instances: legs=$(extrema(legs)), aircraft=$(extrema(immats)), interior_arcs=$(extrema(arcs))"
    end

    dt = compute_normalization(data_train_raw)
    data_train = normalize_data(data_train_raw, dt)
    data_val = normalize_data(data_val_raw, dt)
    data_test = normalize_data(data_test_raw, dt)

    @info "Computing deterministic MIP baseline on test set..."
    t_det = @elapsed det_results = compute_deterministic_baseline(data_test)
    @info "  done in $(round(t_det; digits=1))s"

    @info "Computing deterministic MIP baseline on train/val sets (for the comparison table)..."
    det_results_train = compute_deterministic_baseline(data_train)
    det_results_val = compute_deterministic_baseline(data_val)

    expert_results = expert_baseline(data_test)
    expert_results_train = expert_costs(data_train)
    expert_results_val = expert_costs(data_val)

    (; loss, maximizer) = build_loss(; ε, nb_samples=NB_SAMPLES)

    histories = Dict{Symbol,Any}()
    imitation_results = Dict{Symbol,Vector{NamedTuple}}()
    comparison_rows = Tuple{String,NamedTuple}[]

    for target in TARGETS
        relaxation_flag = target === :relaxation
        @info "Training model for target = $target ($NB_EPOCHS epochs)..."

        Random.seed!(MODEL_SEED)
        model = default_model(nb_features; hidden_dims=HIDDEN_DIMS)

        data_train_t = select_target(data_train, target)
        data_val_t = select_target(data_val, target)

        t_fit = @elapsed history = train_model!(
            model,
            loss,
            maximizer,
            data_train_t,
            data_val_t;
            nb_epochs=NB_EPOCHS,
            relaxation=relaxation_flag,
        )
        @info "  done in $(round(t_fit; digits=1))s (best epoch = $(history.best_epoch))"

        print_loss_curve(target, history)

        # evaluate the checkpoint with the best validation full cost gap, not the last
        # epoch's (possibly overfit) model
        best_model = history.best_model
        histories[target] = history

        @info "Evaluating best model ($target, epoch $(history.best_epoch)) on test set..."
        imitation_results[target] = imitation_baseline(data_test, best_model, maximizer)

        metrics_train = evaluate_metrics(data_train, best_model, maximizer)
        metrics_val = evaluate_metrics(data_val, best_model, maximizer)
        metrics_test = evaluate_metrics(data_test, best_model, maximizer)
        gaps = (;
            train=(;
                op_gap=metrics_train.operational_cost_gap,
                delay_gap=metrics_train.delay_cost_gap,
                full_gap=metrics_train.full_cost_gap,
            ),
            val=(;
                op_gap=metrics_val.operational_cost_gap,
                delay_gap=metrics_val.delay_cost_gap,
                full_gap=metrics_val.full_cost_gap,
            ),
            test=(;
                op_gap=metrics_test.operational_cost_gap,
                delay_gap=metrics_test.delay_cost_gap,
                full_gap=metrics_test.full_cost_gap,
            ),
        )
        push!(comparison_rows, (TARGET_LABEL[target], gaps))
    end

    det_gaps = (;
        train=compute_gaps(det_results_train, expert_results_train),
        val=compute_gaps(det_results_val, expert_results_val),
        test=compute_gaps(det_results, expert_results),
    )
    push!(comparison_rows, ("Deterministic MIP", det_gaps))

    methods = [
        ("Deterministic MIP", det_results),
        (TARGET_LABEL[:integer], imitation_results[:integer]),
        (TARGET_LABEL[:relaxation], imitation_results[:relaxation]),
        ("Expert (diving heuristic)", expert_results),
    ]
    print_summary_table(n_legs, methods, expert_results)
    print_comparison_table(n_legs, comparison_rows)

    return (; det_results, imitation_results, expert_results, histories)
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

for n_legs in NB_LEGS_LIST
    run_benchmark_for_size(n_legs)
end
