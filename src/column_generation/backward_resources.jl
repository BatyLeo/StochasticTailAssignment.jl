"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
struct BackwardResource{P<:PiecewiseLinearFunction}
    "one piecewise linear function per scenario, computing the associated partial path delay cost."
    g::Vector{P}
    "accumulates sum of operational costs"
    operational_cost_sum::Float64
    "accumulated sum of dual variables along the associated partial path."
    λ_sum::Float64
end

"""
$TYPEDSIGNATURES

Compute the cost of combining tail forward path with backward path.
"""
function stochastic_cost(rf::TailForwardResource, rb::BackwardResource)
    cp = partial_cost(rf) + mean(gj(Rj) for (gj, Rj) in zip(rb.g, rf.propagated_delay))
    return cp - rb.λ_sum + rb.operational_cost_sum
end

"""
$TYPEDSIGNATURES

Compute the meet between two backward resource.
"""
function Base.min(r1::BackwardResource, r2::BackwardResource)
    new_g = remove_redundant_breakpoints.(convex_meet.(r1.g, r2.g); atol=1e-8)  # TODO: add an option
    # new_g = min.(r1.g, r2.g)
    new_λ_sum = max(r1.λ_sum, r2.λ_sum)
    new_operational_cost_sum = min(r1.operational_cost_sum, r2.operational_cost_sum)
    return BackwardResource(new_g, new_operational_cost_sum, new_λ_sum)
end

"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
struct PartialBackwardResource
    "accumulates sum of operational costs"
    operational_cost_sum::Float64
    "accumulated sum of dual variables along the associated partial path."
    λ_sum::Float64
end

function BackwardResource(pr::BackwardResource, pr2::PartialBackwardResource)
    return BackwardResource(pr.g, pr2.operational_cost_sum, pr2.λ_sum)
end

"""
$TYPEDSIGNATURES

Compute the meet between two partial backward resource.
"""
function Base.min(r1::PartialBackwardResource, r2::PartialBackwardResource)
    new_λ_sum = max(r1.λ_sum, r2.λ_sum)
    new_operational_cost_sum = min(r1.operational_cost_sum, r2.operational_cost_sum)
    return PartialBackwardResource(new_operational_cost_sum, new_λ_sum)
end
