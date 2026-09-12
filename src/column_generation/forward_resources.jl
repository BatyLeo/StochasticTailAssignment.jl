abstract type AbstractForwardResource end

"""
$TYPEDEF

Forward resource structure for the column generation algorithm, expanded from the tail.

# Fields
$TYPEDFIELDS
"""
struct TailForwardResource <: AbstractForwardResource
    "accumulated sum of operational costs"
    operational_cost_sum::Float64
    "accumulated delay cost"
    delay_cost_sum::Float64
    "accumulated sum of dual variables"
    λ_sum::Float64
    "propagated delay before the current node of the forward path"
    propagated_delay::Vector{Float64}
end

"""
$TYPEDSIGNATURES

Computes the cost of the partial path corresponding to the input forward resource.
"""
function partial_cost(r::TailForwardResource)
    return r.operational_cost_sum + r.delay_cost_sum - r.λ_sum
end

"""
$TYPEDSIGNATURES

Check if forward resource `r₁` dominates forward resource `r₂`.
"""
function Base.:<=(r₁::TailForwardResource, r₂::TailForwardResource)
    # should have lower cost AND delay in order to dominate
    if partial_cost(r₁) > partial_cost(r₂)
        return false
    end
    if any(r₁.propagated_delay .> r₂.propagated_delay)
        return false
    end

    return true
end
