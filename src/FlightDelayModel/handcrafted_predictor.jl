# Reference statistics for a short and medium haul network, used as targets when
# deriving the default `DelayCoefficients` (population LogNormal parameters for the
# departure intrinsic delay before subtracting `dep_shift`, population variance target
# in minutes² for the arrival intrinsic delay before subtracting `arr_shift`, default
# per-airport departure effect in log-minutes for the 8 synthetic airport codes,
# assumed hub transition probability, population hub frequency and duration spread).
const POP_MU_DEP = 3.19
const POP_SIGMA_DEP = 0.491
const POP_VAR_ARR = 84.6
const DEFAULT_AIRPORT_EFFECTS = Dict(
    "HUB" => 0.18,
    "A" => -0.05,
    "B" => 0.10,
    "C" => -0.15,
    "D" => 0.05,
    "E" => -0.12,
    "F" => 0.08,
    "G" => -0.09,
)
const DEFAULT_HUB_FRACTION = 0.85
const POP_HUB_FREQUENCY = DEFAULT_HUB_FRACTION / (1 + DEFAULT_HUB_FRACTION)
const POP_DURATION_SD = 2.56

"""
$TYPEDSIGNATURES

Assumed long-run per-airport usage weights (summing to 1) for `airports`
(`first(airports)` is the hub, every other code an equally likely spoke),
matching the stationary distribution of the random-walk instance generator
(hub -> uniformly random spoke, spoke -> hub with probability
`DEFAULT_HUB_FRACTION`, else another uniformly random spoke). Returns
`Dict(hub => 1.0)` if `airports` has no spoke.

Used to center the per-airport and hub effects of [`DelayCoefficients`](@ref)
so that they average out to `0` on a realistically hub-skewed schedule, rather
than under a naive uniform-over-airports assumption.
"""
function _airport_weights(airports::AbstractVector{String})
    hub = first(airports)
    spokes = @view airports[2:end]
    n = length(spokes)
    n == 0 && return Dict(hub => 1.0)
    f = DEFAULT_HUB_FRACTION
    pi_hub = f / (1 + f)
    pi_spoke = (1 - pi_hub) / n
    return Dict(hub => pi_hub, (s => pi_spoke for s in spokes)...)
end

"""
$TYPEDSIGNATURES

Rescale every value of `d` so that the population standard deviation (no
Bessel correction) of the values equals `target_sd`, preserving the mean (0
for our tables). Returns an all-zero dict if the population standard
deviation of `d` is `0` (a constant table has no direction to rescale along).
"""
function _rescale_to_sd(d::AbstractDict, target_sd::Real)
    vals = collect(values(d))
    m = sum(vals) / length(vals)
    current_sd = sqrt(sum(v -> (v - m)^2, vals) / length(vals))
    current_sd == 0 && return Dict(k => 0.0 for k in keys(d))
    factor = target_sd / current_sd
    return Dict(k => v * factor for (k, v) in d)
end

"""
$TYPEDSIGNATURES

Look up the effect of `code` in table `table`, falling back to a small
deterministic pseudo-random effect for `code` if absent, seeded from `code`
itself so that it stays reproducible, bounded in `[-0.025, 0.025]`, and with
mean zero only in expectation over airport codes (not exactly zero for any
given fallback code).
"""
function _airport_effect(table::AbstractDict, code::AbstractString)
    return get(table, code) do
        h = hash(code)
        return 0.05 * ((h % 1000) / 1000 - 0.5)
    end
end

"""
$TYPEDSIGNATURES

Weighted average of `effect_fn.(keys)` under `weights`.
"""
function _weighted_mean(weights::AbstractDict, keys, effect_fn)
    return sum(k -> weights[k] * effect_fn(k), keys)
end

"""
$TYPEDEF

Named per-leg effects making up one weight vector (`mu_dep`, `sigma_dep`,
`mu_arr` or `sigma_arr`) of [`DelayCoefficients`](@ref), all expressed
relative to the population baseline of that component (`base`, or the
`tau`/`population_variance` heterogeneity identity for the sigma components)
and mean-centered by [`compile_weights`](@ref) so that they only redistribute
delay across legs, without shifting the population mean.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DelayEffects
    "population baseline of the component (component units, pre-activation for mu_dep and sigma_dep/sigma_arr)"
    base::Float64 = 0.0
    "hub effect amplitude (component units)"
    hub::Float64 = 0.0
    "own-airport effect amplitude, applied to the per-airport effect table (component units)"
    airport::Float64 = 0.0
    "other-airport (cross) effect amplitude, applied to the per-airport effect table (component units)"
    cross_airport::Float64 = 0.0
    "hour-of-day effect for the 00h-05h, 06h-09h, 10h-13h, 14h-17h and 18h-23h bands, in order (component units)"
    hours::NTuple{5,Float64} = ntuple(_ -> 0.0, 5)
    "day-of-week effect, Monday to Sunday, in order (component units)"
    days::NTuple{7,Float64} = ntuple(_ -> 0.0, 7)
    "duration effect, per hour of deviation from a 5h reference duration (component units)"
    duration::Float64 = 0.0
    "across-leg heterogeneity standard deviation of the mean, measured at risk_spread = 1 (component units, sigma components only)"
    tau::Float64 = 0.0
    "population variance target used to re-derive the within-leg sigma base (component units squared, sigma components only)"
    population_variance::Float64 = 0.0
end

"""
$TYPEDEF

Handcrafted coefficients parameterizing the intrinsic (root) delay
distributions of [`SyntheticDelayModel`](@ref): one [`DelayEffects`](@ref) per
weight vector of a [`HandcraftedDelayPredictor`](@ref), a shared per-airport
effect table, and the dynamic-simulator-only `beta_*` coefficients.

The `beta_*_propagated`, `beta_dep_has_propagated` and `beta_arr_has_departure_delay`
coefficients are only used by the dynamic simulator ([`propagate_delays`](@ref) for `SyntheticDelayModel`),
never by [`sample_root_scenarios`](@ref). They default to `0.0`, meaning the
root delay model is purely static (route-independent). Setting them to nonzero
values makes the resulting delays route-dependent, which is only valid for
simulation, not as an input to `stochastic_column_generation`.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DelayCoefficients
    "departure mu effects, pre-activation (log-minutes)"
    mu_dep::DelayEffects = DelayEffects(;
        base=inv_softplus(POP_MU_DEP),
        hub=0.20,
        airport=1.0,
        cross_airport=0.5,
        hours=(-0.10, -0.20, 0.00, 0.15, 0.25),
        days=(0.06, 0.00, 0.00, 0.04, 0.10, -0.10, -0.02),
        duration=0.04,
    )
    "departure sigma effects, pre-activation (log-minutes)"
    sigma_dep::DelayEffects = DelayEffects(;
        hub=0.08, duration=0.03, tau=0.265, population_variance=POP_SIGMA_DEP^2
    )
    "arrival mu effects (minutes)"
    mu_arr::DelayEffects = DelayEffects(;
        base=42.0 - 6.7,
        hub=1.5,
        airport=1.0,
        hours=(-0.5, -1.0, 0.0, 0.8, 1.2),
        days=(0.12, 0.00, 0.00, 0.08, 0.20, -0.20, -0.04),
        duration=-0.8,
    )
    "arrival sigma effects, pre-activation (minutes)"
    sigma_arr::DelayEffects = DelayEffects(;
        duration=0.8, tau=3.0, population_variance=POP_VAR_ARR
    )
    "per-airport departure effect table (log-minutes, mean 0)"
    airport_effects::Dict{String,Float64} = DEFAULT_AIRPORT_EFFECTS
    "target standard deviation of the per-airport arrival effect table (minutes)"
    airport_arr_target_sd::Float64 = 1.2

    "departure mu effect of a nonzero upstream propagated delay (dynamic simulator only)"
    beta_dep_has_propagated::Float64 = 0.0
    "departure mu effect of the upstream propagated delay value, per hour (dynamic simulator only)"
    beta_dep_propagated::Float64 = 0.0
    "arrival mu effect of a nonzero departure delay (dynamic simulator only)"
    beta_arr_has_departure_delay::Float64 = 0.0
    "arrival mu effect of the departure delay value, per hour (dynamic simulator only)"
    beta_arr_propagated::Float64 = 0.0
end

function _hour_band_effect(h::Integer, hours::NTuple{5,Float64})
    h <= 5 && return hours[1]
    h <= 9 && return hours[2]
    h <= 13 && return hours[3]
    h <= 17 && return hours[4]
    return hours[5]
end

"""
$TYPEDSIGNATURES

Approximate the across-leg variance of the (linearized) within-leg sigma
induced by the hub and duration effects of `effects`, assuming the underlying
features are mutually independent: the hub/spoke indicator (Bernoulli, mean
`POP_HUB_FREQUENCY`) and the leg duration (standard deviation
`POP_DURATION_SD` hours). Both effects are scaled by `risk_spread`, matching
how they enter the weight vector in [`compile_weights`](@ref).
"""
function _sigma_legs_variance(effects::DelayEffects, risk_spread::Real)
    r = risk_spread
    hub_variance = POP_HUB_FREQUENCY * (1 - POP_HUB_FREQUENCY)
    return (r * effects.hub)^2 * hub_variance + (r * effects.duration)^2 * POP_DURATION_SD^2
end

"""
$TYPEDSIGNATURES

Population baseline (intercept, pre-activation) of one weight vector compiled
by [`compile_weights`](@ref) for `component` (`:mu_dep`, `:sigma_dep`,
`:mu_arr` or `:sigma_arr`).

For the two mu components, this is simply `effects.base` (`risk_spread`
does not apply to the population baseline itself, only to the per-leg
effects around it). For the two sigma components, `effects.base` is unused:
the baseline is instead re-derived from the law-of-total-variance identity
`(risk_spread * effects.tau)^2 + sigma_base^2 + sigma_legs_variance =
effects.population_variance` (see [`_sigma_legs_variance`](@ref)), erroring
out with a clear message if `risk_spread` is so large that the heterogeneity
and sigma-variance terms alone would reach or exceed the population variance
target.
"""
function _base_weight(effects::DelayEffects, risk_spread::Real, component::Symbol)
    component in (:mu_dep, :mu_arr) && return effects.base

    r = risk_spread
    sigma_legs_variance = _sigma_legs_variance(effects, r)
    radicand = effects.population_variance - (r * effects.tau)^2 - sigma_legs_variance
    if radicand <= 0
        error(
            "risk_spread=$r is too large for the $component component: the " *
            "heterogeneity term (risk_spread * tau)^2 = $((r * effects.tau)^2) plus the " *
            "across-leg sigma variance $sigma_legs_variance would reach or exceed the " *
            "population variance target $(effects.population_variance). Use a smaller risk_spread.",
        )
    end
    return inv_softplus(sqrt(radicand))
end

"""
$TYPEDEF

Linear, handcrafted predictor for the intrinsic (root) delay distribution
parameters (see `departure_prediction` and `arrival_prediction`), with weights
compiled once from named, interpretable [`DelayCoefficients`](@ref) rather than
learned.

Since one-hot feature blocks always sum to 1 for a given leg, population
baselines are injected as a uniform addition over the `DayOfWeek` one-hot block
of the relevant weight vector (see [`compile_weights`](@ref)): this acts as an
intercept without requiring a dedicated bias feature.

# Fields
$TYPEDFIELDS
"""
@kwdef struct HandcraftedDelayPredictor
    "weight vector for the departure mu parameter, pre-activation"
    w_mu_dep::Vector{Float32}
    "weight vector for the departure sigma parameter, pre-activation"
    w_sigma_dep::Vector{Float32}
    "weight vector for the arrival mu parameter"
    w_mu_arr::Vector{Float32}
    "weight vector for the arrival sigma parameter, pre-activation"
    w_sigma_arr::Vector{Float32}
    "names of the static features, aligned with the weight vectors"
    feature_names::Vector{String}
    "coefficients used to compile the weight vectors"
    coefficients::DelayCoefficients
end

"""
$TYPEDSIGNATURES

Add `value` to every entry of the `DayOfWeek` one-hot block of `w`, indexed via
`index`. Since exactly one such entry is active (equal to 1) for any leg, this
acts as an intercept term.
"""
function _add_intercept!(w::Vector{Float32}, index::Dict{String,Int}, days, value::Real)
    for d in days
        w[index["DayOfWeek/$d"]] += Float32(value)
    end
    return w
end

"""
$TYPEDSIGNATURES

Write the mean-centered effect `effect_fn.(values)`, scaled by `risk_spread`,
into the one-hot block `"\$prefix/\$v"` of `w` for each `v` in `values`.
"""
function _fill_onehot_mu!(
    w::Vector{Float32}, index::Dict{String,Int}, prefix::String, values, effect_fn, r::Real
)
    mean_val = sum(effect_fn, values) / length(values)
    for v in values
        w[index["$prefix/$v"]] = Float32(r * (effect_fn(v) - mean_val))
    end
    return w
end

"""
$TYPEDSIGNATURES

Feature-name layout matching `component` (`:mu_dep`/`:sigma_dep` for the
departure layout, `:mu_arr`/`:sigma_arr` for the mirrored arrival layout),
used by [`compile_weights`](@ref) to look up the right one-hot blocks of
`config`'s feature vector: `hour_prefix`/`hours` for the own hour-of-day
block, `airport_prefix` for the own airport block, `cross_prefix` for the
other leg endpoint's airport block.
"""
function _component_layout(config::FeaturesConfig, component::Symbol)
    if component in (:mu_dep, :sigma_dep)
        return (;
            hour_prefix="DepartureHour",
            hours=config.start_hours,
            airport_prefix="DepartureAirport",
            cross_prefix="ArrivalAirport",
        )
    else
        return (;
            hour_prefix="ArrivalHour",
            hours=config.end_hours,
            airport_prefix="ArrivalAirport",
            cross_prefix="DepartureAirport",
        )
    end
end

"""
$TYPEDSIGNATURES

Compile the weight vector for one component (`:mu_dep`, `:sigma_dep`,
`:mu_arr` or `:sigma_arr`) of a [`HandcraftedDelayPredictor`](@ref) from
`effects` and `config`, scaling every named per-leg effect of `effects` by
`risk_spread`. `airport_effects` is the per-airport effect table read for both
the own-airport and cross-airport (other leg endpoint) blocks.
"""
function compile_weights(
    effects::DelayEffects,
    config::FeaturesConfig,
    component::Symbol;
    airport_effects::AbstractDict,
    risk_spread::Real=1.0,
)
    (; hour_prefix, hours, airport_prefix, cross_prefix) = _component_layout(
        config, component
    )
    (; index) = compute_feature_info(config)
    r = risk_spread
    w = zeros(Float32, length(index))

    _fill_onehot_mu!(
        w, index, hour_prefix, hours, h -> _hour_band_effect(h, effects.hours), r
    )
    _fill_onehot_mu!(w, index, "DayOfWeek", config.days, d -> effects.days[d], r)

    # per-airport and hub effects are centered on the hub-skewed usage
    # distribution (`_airport_weights`) rather than a naive uniform-over-airports
    # average, so that they still average out to 0 on a schedule generated with
    # the default hub_fraction (hour and day effects above are centered
    # uniformly, since their categories are visited with equal long-run
    # frequency by construction)
    hub = first(config.airports)
    usage_weights = _airport_weights(config.airports)
    mean_airport = _weighted_mean(
        usage_weights, config.airports, a -> _airport_effect(airport_effects, a)
    )
    mean_hub = usage_weights[hub]
    for a in config.airports
        centered_airport = _airport_effect(airport_effects, a) - mean_airport
        centered_hub = (a == hub ? 1.0 : 0.0) - mean_hub
        w[index["$airport_prefix/$a"]] += Float32(
            r * (effects.airport * centered_airport + effects.hub * centered_hub)
        )
        w[index["$cross_prefix/$a"]] += Float32(
            r * effects.cross_airport * centered_airport
        )
        # the "/other" airport features (for codes outside `config.airports`)
        # keep weight 0
    end

    w[index["Duration"]] += Float32(r * effects.duration)

    _add_intercept!(
        w,
        index,
        config.days,
        _base_weight(effects, r, component) - 5 * r * effects.duration,
    )

    return w
end

"""
$TYPEDSIGNATURES

Build a [`HandcraftedDelayPredictor`](@ref) from `coefficients` and `config`.
"""
function HandcraftedDelayPredictor(
    coefficients::DelayCoefficients, config::FeaturesConfig; risk_spread::Real=1.0
)
    (; airport_effects, airport_arr_target_sd) = coefficients
    arrival_airport_effects = _rescale_to_sd(airport_effects, airport_arr_target_sd)

    w_mu_dep = compile_weights(
        coefficients.mu_dep, config, :mu_dep; airport_effects, risk_spread
    )
    w_sigma_dep = compile_weights(
        coefficients.sigma_dep, config, :sigma_dep; airport_effects, risk_spread
    )
    w_mu_arr = compile_weights(
        coefficients.mu_arr,
        config,
        :mu_arr;
        airport_effects=arrival_airport_effects,
        risk_spread,
    )
    w_sigma_arr = compile_weights(
        coefficients.sigma_arr, config, :sigma_arr; airport_effects, risk_spread
    )

    return HandcraftedDelayPredictor(;
        w_mu_dep,
        w_sigma_dep,
        w_mu_arr,
        w_sigma_arr,
        feature_names=config.names,
        coefficients,
    )
end

"""
$TYPEDSIGNATURES

Predict the departure intrinsic delay LogNormal parameters from static
features `x_dep`.

`mu_offset` is added to the pre-activation `mu_dep` linear combination before the
`softplus`, used by [`propagate_delays`](@ref) to inject the dynamic
(route-dependent) `beta_dep_has_propagated`/`beta_dep_propagated` effects without
duplicating the weight dot product.
"""
function departure_prediction(p::HandcraftedDelayPredictor, x_dep; mu_offset::Real=0.0)
    return (;
        mu_dep=softplus(dot(p.w_mu_dep, x_dep) + mu_offset),
        sigma_dep=softplus(dot(p.w_sigma_dep, x_dep)),
    )
end

"""
$TYPEDSIGNATURES

Predict the arrival intrinsic delay Normal parameters from static features
`x_arr`.

`mu_offset` is added to the `mu_arr` linear combination, used by
[`propagate_delays`](@ref) to inject the dynamic (route-dependent)
`beta_arr_has_departure_delay`/`beta_arr_propagated` effects without duplicating the
weight dot product.
"""
function arrival_prediction(p::HandcraftedDelayPredictor, x_arr; mu_offset::Real=0.0)
    return (;
        mu_arr=dot(p.w_mu_arr, x_arr) + mu_offset,
        sigma_arr=softplus(dot(p.w_sigma_arr, x_arr)),
    )
end

"""
$TYPEDSIGNATURES

Return `feature_name => weight` pairs for each of the 4 weight vectors of
`predictor`, as a named tuple `(; mu_dep, sigma_dep, mu_arr, sigma_arr)`.
"""
function weight_table(predictor::HandcraftedDelayPredictor)
    names = predictor.feature_names
    return (;
        mu_dep=Pair.(names, predictor.w_mu_dep),
        sigma_dep=Pair.(names, predictor.w_sigma_dep),
        mu_arr=Pair.(names, predictor.w_mu_arr),
        sigma_arr=Pair.(names, predictor.w_sigma_arr),
    )
end
