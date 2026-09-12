"""
$(TYPEDSIGNATURES)

Generate an `nb_scenarios x nb_legs(schedule)` matrix of root delays (in minutes),
compatible with [`propagate_delays_from_root_delays`](@ref) and
`full_cost(routes, root_delays, instance; delay_cost_function)`.

For each leg, the root delay in a given scenario is the sum of an independent
departure intrinsic delay and an independent arrival intrinsic delay, each drawn
from a `LogNormal(mu, sigma)` distribution. The `mu` parameter is increased by
`evening_mu_boost` for legs departing in the evening or at night (hour `>= 17` or
`<= 5`), to mimic the pattern where evening flights tend to accumulate more delay.

This is a purely parametric sampler (no ML model, no features), meant to produce
plausible-looking delay scenarios for synthetic instances.
"""
function generate_root_delays(
    schedule::ActivitySchedule;
    nb_scenarios::Int=50,
    base_mu::Float64=2.0,
    base_sigma::Float64=0.8,
    evening_mu_boost::Float64=0.5,
    seed::Int=0,
)
    rng = Random.Xoshiro(seed)
    L = nb_legs(schedule)

    root_delays = zeros(Float32, nb_scenarios, L)

    for l in 1:L
        leg = schedule.legs[l]
        hour = Dates.hour(departure_time(leg))

        mu = base_mu
        if hour >= 17 || hour <= 5
            mu += evening_mu_boost
        end

        dist = LogNormal(mu, base_sigma)

        for s in 1:nb_scenarios
            departure_delay = rand(rng, dist)
            arrival_delay = rand(rng, dist)
            root_delays[s, l] = Float32(departure_delay + arrival_delay)
        end
    end

    return root_delays
end
