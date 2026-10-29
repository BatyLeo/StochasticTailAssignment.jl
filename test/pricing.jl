@testitem "Backward-bound pricing matches brute-force enumeration" begin
    using StochasticTailAssignment.AircraftRoutingBase
    using StochasticTailAssignment.FlightDelayModel
    using Graphs: outneighbors

    function leg(id, from, to, dep, arr)
        return Leg(;
            id,
            departure_airport=from,
            arrival_airport=to,
            departure_time=string_to_date("2022-01-01T$dep"),
            arrival_time=string_to_date("2022-01-01T$arr"),
            aircraft_type="77W",
            minimum_turn_time=0,
        )
    end

    # Legs 1 and 2 are alternatives before the maintenance (index 6), legs 3 and 4 are
    # alternatives after it, and leg 5 can follow either of them.
    # Leg 1 reaches the maintenance with a 30 minutes slack, leg 2 with a 90 minutes slack.
    # The maintenance is chained to leg 3 with no slack, so a delay propagated into the
    # maintenance is passed on to leg 3 (but is not charged at the maintenance itself).
    schedule = ActivitySchedule(;
        immats=[Immat("A", 1.0, "s", "77W")],
        standard_consumption=Dict("77W" => 125.0, "" => 65.0),
        legs=[
            leg("1", "JFK", "CDG", "00:00", "06:00"),
            leg("2", "JFK", "CDG", "01:00", "05:00"),
            leg("3", "CDG", "JFK", "08:00", "14:00"),
            leg("4", "CDG", "JFK", "10:00", "16:00"),
            leg("5", "JFK", "CDG", "17:00", "23:00"),
        ],
        maintenances=[
            Maintenance(;
                id="M",
                location="CDG",
                start_time=string_to_date("2022-01-01T06:30"),
                end_time=string_to_date("2022-01-01T08:00"),
                immat="A",
                minimum_turn_time=0,
            ),
        ],
        TTM_factor=1.0,
    )
    maintenance_index = 6
    @test is_maintenance(schedule, maintenance_index)

    # root_delays[s, l] is the delay of leg l in scenario s
    root_delays = zeros(4, 5)
    root_delays[:, 1] = [0.0, 60.0, 90.0, 120.0]
    root_delays[:, 3] = [0.0, 0.0, 10.0, 0.0]
    root_delays[:, 4] = [0.0, 5.0, 0.0, 0.0]
    delay_cost_function = DelayCostFunction(;
        breakpoints=[15.0, 60.0], slopes=[0.0, 1.0, 3.0]
    )

    # all s-t paths of the aircraft graph, as routes (without s and t)
    graph = schedule.immat_graphs[1]
    s, t = get_s(schedule), get_t(schedule)
    function enumerate_paths(u, prefix)
        u == t && return [prefix]
        return reduce(
            vcat,
            [
                enumerate_paths(v, v == t ? prefix : [prefix; v]) for
                v in outneighbors(graph, u)
            ];
            init=Vector{Int}[],
        )
    end
    paths = enumerate_paths(s, Int[])
    @test length(paths) == 15
    @test all(p -> maintenance_index in p, paths)

    partial_bounds = StochasticTailAssignment.compute_partial_bounds(
        schedule, root_delays; delay_cost_function
    )
    orders = StochasticTailAssignment._precompute_topological_orders(schedule)
    (; FF, BF) = StochasticTailAssignment._precompute_extension_functions(
        schedule, 1, root_delays
    )

    # With λ[1] = 440, the best route is leg 1 then the maintenance (cost -23.75), while the
    # route made of the maintenance only (cost 0) is found first. The tight connection
    # leaves a positive delay at the maintenance in some scenarios, so a backward bound that
    # charges a delay cost there is not a lower bound anymore and prunes the best route.
    for λ_val in (
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [440.0, 0.0, 0.0, 0.0, 0.0],
        [300.0, 0.0, 50.0, 0.0, 10.0],
    )
        brute_force = minimum(paths) do p
            route = Route(1, p)
            legs = leg_indices_from_route(route, schedule)
            return full_cost(route, root_delays, schedule; delay_cost_function) -
                   sum(λ_val[l] for l in legs; init=0.0)
        end
        (; p_star, c_star) = backward_tail_shortest_path(
            schedule,
            1,
            λ_val,
            root_delays,
            partial_bounds[1],
            deepcopy(partial_bounds[1]),
            FF,
            BF,
            orders[1];
            delay_cost_function,
        )
        @test c_star ≈ brute_force atol = 1e-3
        route_star = Route(1, p_star)
        @test full_cost(route_star, root_delays, schedule; delay_cost_function) - sum(
            λ_val[l] for l in leg_indices_from_route(route_star, schedule); init=0.0
        ) ≈ c_star atol = 1e-3
    end
end
