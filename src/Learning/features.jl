"""
$TYPEDSIGNATURES

Compute learning features for each interior arc (arc not connected to the source `s`
nor the sink `t`) of `instance`, using the unmerged root delay scenarios
`departure_root_delays` and `arrival_root_delays`.

`departure_root_delays` and `arrival_root_delays` are `S x L` matrices (`S` scenarios,
`L` legs), as produced by [`sample_root_scenarios_unmerged`](@ref) (their sum matches the
merged `root_delays` used for the expert labels and evaluation, see
[`generate_dataset`](@ref)). Column `j` of each matrix corresponds to `instance.legs[j]`
(for an `ActivitySchedule`, leg vertex indices are exactly `1:nb_legs(instance)`, so a leg
vertex `v` is also its own column index).

For each interior arc `(u, v, i)`, the following 23 features are computed, following the
slack definition of the delay-model-informed features (the downstream leg `v`'s intrinsic
delay only affects its own departure, while the upstream leg `u` contributes its arrival
intrinsic delay):
1. `ω`, the slack between `u` and `v` (1 value)
2. `ω_tt`, the slack between `u` and `v` including the minimum turn time (1 value)
3. `quantiles_with`, the `0.1:0.1:1.0` quantiles of the slack scenarios
   `ω_tt - eps_arr(u) + eps_dep(v)`, accounting for the departure intrinsic delay of `v`
   (10 values)
4. `quantiles_without`, the `0.1:0.1:1.0` quantiles of the slack scenarios
   `ω_tt - eps_arr(u)`, without accounting for the departure intrinsic delay of `v`
   (10 values)
5. `cost`, the edge cost of `(u, v, i)` (1 value)

Returns a `Matrix{Float32}` of size `(23, nb_interior_arcs(instance))`, where column
`arc_index[u, v, i]` holds the features of arc `(u, v, i)`.
"""
function compute_features(
    instance::ActivitySchedule,
    departure_root_delays::AbstractMatrix,
    arrival_root_delays::AbstractMatrix,
)
    (; arc_index, nb_interior_arcs, immat_graphs) = instance
    arc_index === nothing && error(
        "compute_features requires an ActivitySchedule built with store_arc_index=true"
    )
    S = size(departure_root_delays, 1)
    I = nb_immats(instance)
    p = 0.1:0.1:1.0

    nb_features = 23  # 2 slacks + 10 quantiles_with + 10 quantiles_without + 1 cost
    features = zeros(Float32, nb_features, nb_interior_arcs)
    zero_delays = zeros(Float32, S)

    for i in 1:I
        for arc in edges(immat_graphs[i])
            u, v = src(arc), dst(arc)
            # Skip arcs connected to source or sink
            if is_s(instance, u) ||
                is_t(instance, u) ||
                is_s(instance, v) ||
                is_t(instance, v)
                continue
            end
            idx = arc_index[u, v, i]

            act_u = get_activity(instance, u)
            act_v = get_activity(instance, v)
            ω = Float32(slack(act_u, act_v))
            ω_tt = Float32(slack_with_turn_time(act_u, act_v))

            # leg vertex indices double as their column index in the root delay
            # matrices (see `leg_indices(::ActivitySchedule)`)
            u_arrival_delays =
                is_leg(instance, u) ? view(arrival_root_delays, :, u) : zero_delays
            v_departure_delays =
                is_leg(instance, v) ? view(departure_root_delays, :, v) : zero_delays

            slack_without_v = ω_tt .- u_arrival_delays
            slack_with_v = slack_without_v .+ v_departure_delays

            features[1, idx] = ω
            features[2, idx] = ω_tt
            features[3:12, idx] .= quantile(slack_with_v, p)
            features[13:22, idx] .= quantile(slack_without_v, p)
            features[23, idx] = Float32(edge_cost(instance, u, v, i))
        end
    end

    return features
end
