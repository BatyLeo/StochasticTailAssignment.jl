"""
    benchmark_tables.jl

Reproduce the benchmark result tables from the instance generator documentation.
Run from the package root: `julia --project=scripts scripts/benchmark_tables.jl`

Tables produced:
1. Deterministic vs. stochastic gap (varying instance size)
2. Seed stability (varying random seed at fixed size)
3. Structural statistics and scaling
"""

using StochasticTailAssignment
using StochasticTailAssignment.AircraftRoutingBase
using StochasticTailAssignment.InstanceGenerator
using StochasticTailAssignment.FlightDelayModel
using Graphs
using Printf
using Dates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function run_pipeline(schedule, root_delays, delay_cost_fn; silent=true)
    det_routes, _, _ = solve_aircraft_routing(schedule)
    @assert is_feasible(det_routes, schedule) "Deterministic solution infeasible"

    det_op = operational_cost(det_routes, schedule)
    det_delay = delay_expected_cost(
        det_routes, root_delays, schedule; delay_cost_function=delay_cost_fn
    )
    det_total = det_op + det_delay

    t_cg = @elapsed cg = stochastic_column_generation(
        schedule,
        det_routes;
        model_builder=highs_model,
        root_delays,
        delay_cost_function=delay_cost_fn,
        max_nb_columns=10000,
        tol=1e-6,
        silent=true,
    )
    @assert cg.feasible "Column generation infeasible"

    t_div = @elapsed diving, div_feasible = diving_heuristic_with_backtracking!(
        schedule,
        cg.columns,
        root_delays,
        cg.dual_values,
        3;
        model_builder=highs_model,
        delay_cost_function=delay_cost_fn,
        silent=silent,
    )

    if !div_feasible || !is_feasible(diving, schedule)
        return (;
            det_op,
            det_delay,
            det_total,
            nb_columns=length(cg.columns),
            cg_bound=cg.obj,
            time_cg=t_cg,
            time_div=t_div,
            feasible=false,
        )
    end

    sto_op = operational_cost(diving, schedule)
    sto_delay = delay_expected_cost(
        diving, root_delays, schedule; delay_cost_function=delay_cost_fn
    )
    sto_total = sto_op + sto_delay

    route_lengths = sort([length(leg_indices_from_route(r, schedule)) for r in diving])

    return (;
        det_op,
        det_delay,
        det_total,
        sto_op,
        sto_delay,
        sto_total,
        nb_columns=length(cg.columns),
        cg_bound=cg.obj,
        route_lengths,
        det_routes,
        diving,
        time_cg=t_cg,
        time_div=t_div,
        feasible=true,
    )
end

pct(new, old) = 100.0 * (new - old) / abs(old)

# ---------------------------------------------------------------------------
# Table 1: Deterministic vs. stochastic gap
# ---------------------------------------------------------------------------
println("=" ^ 80)
println("TABLE 1: Deterministic vs. stochastic gap")
println("=" ^ 80)
println()

gap_configs = [
    (nb_legs=50, nb_scenarios=50, seed=42),
    (nb_legs=80, nb_scenarios=50, seed=42),
    (nb_legs=100, nb_scenarios=50, seed=42),
]

@printf(
    "%-14s  %10s  %12s  %12s  %8s  %s\n",
    "Instance",
    "Δ op cost",
    "Δ delay cost",
    "Δ total cost",
    "Columns",
    "Route lengths"
)
println("-" ^ 85)

for cfg in gap_configs
    schedule, root_delays, delay_cost_fn = generate_benchmark_instance(
        cfg.nb_legs; nb_scenarios=cfg.nb_scenarios, seed=cfg.seed
    )
    I = nb_immats(schedule)

    print("  Running $(cfg.nb_legs)L / $(I)ac ... ")
    t = @elapsed res = run_pipeline(schedule, root_delays, delay_cost_fn)
    if !res.feasible
        println("INFEASIBLE ($(round(t; digits=1))s)")
        @printf(
            "%-14s  %10s  %12s  %12s  %8d  %s\n",
            "$(cfg.nb_legs)L / $(I)ac",
            "N/A",
            "N/A",
            "N/A",
            res.nb_columns,
            "N/A"
        )
        continue
    end
    println("done ($(round(t; digits=1))s)")

    @printf(
        "%-14s  %+9.1f%%  %+11.1f%%  %+11.1f%%  %8d  %s\n",
        "$(cfg.nb_legs)L / $(I)ac",
        pct(res.sto_op, res.det_op),
        pct(res.sto_delay, res.det_delay),
        pct(res.sto_total, res.det_total),
        res.nb_columns,
        res.route_lengths,
    )
end

println()

# ---------------------------------------------------------------------------
# Table 2: Seed stability
# ---------------------------------------------------------------------------
println("=" ^ 80)
println("TABLE 2: Seed stability (50L, default aircraft count)")
println("=" ^ 80)
println()

@printf("%-6s  %10s  %12s  %12s\n", "Seed", "Δ op cost", "Δ delay cost", "Δ total cost")
println("-" ^ 50)

for seed in [0, 1, 42]
    schedule, root_delays, delay_cost_fn = generate_benchmark_instance(
        50; nb_scenarios=50, seed
    )
    I = nb_immats(schedule)

    print("  Running seed=$seed ($(I) aircraft) ... ")
    t = @elapsed res = run_pipeline(schedule, root_delays, delay_cost_fn)
    if !res.feasible
        println("INFEASIBLE ($(round(t; digits=1))s)")
        @printf("%-6d  %10s  %12s  %12s\n", seed, "N/A", "N/A", "N/A")
        continue
    end
    println("done ($(round(t; digits=1))s)")

    @printf(
        "%-6d  %+9.1f%%  %+11.1f%%  %+11.1f%%\n",
        seed,
        pct(res.sto_op, res.det_op),
        pct(res.sto_delay, res.det_delay),
        pct(res.sto_total, res.det_total),
    )
end

println()

# ---------------------------------------------------------------------------
# Table 3: Structural statistics and scaling
# ---------------------------------------------------------------------------
println("=" ^ 80)
println("TABLE 3: Structural statistics and scaling")
println("=" ^ 80)
println()

@printf(
    "%-14s  %8s  %8s  %10s  %10s  %s\n",
    "Instance",
    "Legs/ac",
    "Gen (s)",
    "Det (s)",
    "E/L²",
    "Route lengths"
)
println("-" ^ 75)

for nb_legs_val in [50, 100, 200, 300]
    t_gen = @elapsed begin
        schedule, _, _ = generate_benchmark_instance(nb_legs_val; seed=42)
    end
    L = nb_legs(schedule)
    I = nb_immats(schedule)
    E = ne(schedule)
    edge_density = E / L^2

    t_det = @elapsed begin
        det_routes, _, _ = solve_aircraft_routing(schedule)
    end

    route_lens = sort([length(leg_indices_from_route(r, schedule)) for r in det_routes])

    @printf(
        "%-14s  %8.1f  %8.1f  %10.1f  %10.3f  [%d..%d]\n",
        "$(L)L / $(I)ac",
        L / I,
        t_gen,
        t_det,
        edge_density,
        minimum(route_lens),
        maximum(route_lens),
    )
end
