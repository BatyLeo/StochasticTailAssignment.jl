"""
$(TYPEDSIGNATURES)

Build a [`SyntheticDelayModel`](@ref) for `schedule`, using a
[`HandcraftedDelayPredictor`](@ref) compiled from `coefficients` (pass a
custom per-airport effect table via `coefficients=DelayCoefficients(;
airport_effects=...)`).

`delay_intensity` globally scales every intrinsic root delay after shift and
cap (see [`SyntheticDelayModel`](@ref)). `risk_spread` scales every named
per-leg effect of `coefficients` and re-derives the within-leg sigma so that
the population-level delay distribution stays approximately unchanged while
the spread of delay risk across legs increases (see [`DelayCoefficients`](@ref)).

The returned model is static-only (`has_dynamic_features(model) == false`), so
it is a valid input to [`sample_root_scenarios`](@ref) and, through
[`generate_root_delays`](@ref), to `stochastic_column_generation`.
"""
function build_delay_model(
    schedule::ActivitySchedule;
    coefficients::DelayCoefficients=DelayCoefficients(),
    delay_intensity::Real=1.0,
    risk_spread::Real=1.0,
    config::FeaturesConfig=FeaturesConfig(; airports=schedule_airports(schedule)),
)
    predictor = HandcraftedDelayPredictor(coefficients, config; risk_spread)
    return SyntheticDelayModel(; predictor, delay_intensity)
end

"""
$(TYPEDSIGNATURES)

Generate an `nb_scenarios x nb_legs(schedule)` matrix of root delays (in
minutes), compatible with [`propagate_delays_from_root_delays`](@ref) and
`full_cost(routes, root_delays, instance; delay_cost_function)`.

Delays are generated from a [`SyntheticDelayModel`](@ref) (see
[`build_delay_model`](@ref)): for each leg, the root delay in a given scenario
is the sum of a departure intrinsic delay (LogNormal) and an arrival intrinsic
delay (Normal), whose parameters depend on handcrafted, interpretable static
features of the leg (hub/spoke airports, day of week, hour of day, duration).
Negative root delays are not clamped to zero, since they represent early legs
and are handled correctly by the delay propagation equations (they provide
extra slack absorption).

See [`DelayCoefficients`](@ref) for the default model parameters, and
`delay_intensity`/`risk_spread` in [`build_delay_model`](@ref) for the two
global knobs exposed here.
"""
function generate_root_delays(
    schedule::ActivitySchedule;
    nb_scenarios::Int=50,
    seed::Int=0,
    delay_intensity::Real=1.0,
    risk_spread::Real=1.0,
    coefficients::DelayCoefficients=DelayCoefficients(),
)
    config = FeaturesConfig(; airports=schedule_airports(schedule))
    model = build_delay_model(schedule; coefficients, delay_intensity, risk_spread, config)
    scenarios = DelayScenarios(schedule; nb_scenarios, config, seed)
    return sample_root_scenarios(model, scenarios)
end
