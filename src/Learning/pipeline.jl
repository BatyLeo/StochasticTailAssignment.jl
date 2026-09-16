"""
$TYPEDSIGNATURES

Build the combinatorial optimization maximizer used as the pricing oracle in the
InferOpt pipeline.

The returned function takes the predicted edge weights `θ` (one per interior arc) and
solves the edge-based MIP (see [`aircraft_routing_edge_maximizer`](@ref)) to return the
corresponding arc solution.
"""
function build_maximizer(; model_builder=highs_model)
    function maximizer(θ; instance, silent=true, relaxation=false)
        return aircraft_routing_edge_maximizer(
            θ;
            instance,
            include_chaining_costs=true,
            model_builder,
            use_operational_costs=false,
            silent,
            relaxation,
        )
    end
    return maximizer
end

"""
$TYPEDSIGNATURES

Build the Fenchel-Young loss (with a `PerturbedAdditive` optimization layer) used to
train the pricing model by imitation of the diving heuristic's expert solutions.

Returns a `NamedTuple` with fields `loss` (the `FenchelYoungLoss`) and `maximizer` (the
raw combinatorial oracle, see [`build_maximizer`](@ref)).
"""
function build_loss(; ε=0.01, nb_samples=10, model_builder=highs_model)
    maximizer = build_maximizer(; model_builder)
    function g(y; instance, kwargs...)
        return y[1:(instance.nb_interior_arcs)]
    end
    gm_layer = LinearMaximizer(maximizer; g)
    perturbed = PerturbedAdditive(gm_layer; ε, nb_samples)
    fyl = FenchelYoungLoss(perturbed)
    return (; loss=fyl, maximizer)
end
