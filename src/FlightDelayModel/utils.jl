"""
$TYPEDSIGNATURES

Numerically stable softplus activation, `log(1 + exp(x))`.
"""
softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))

"""
$TYPEDSIGNATURES

Inverse of [`softplus`](@ref), `log(exp(y) - 1)`.
"""
inv_softplus(y) = log(expm1(y))
