const DELAY_COST_BREAKPOINTS = Float64[60.0, 165.0]

# These slopes are expressed per seat per minute of delay, and are multiplied by
# a single fleet-wide seat count to get the actual cost slopes (see
# `generate_benchmark_instance`'s `nb_seats` keyword).
const DELAY_COST_SLOPE_MEDIUM_HAUL = [0.09, 0.34, 1.34]

"""
Small positive shift added to the standard deviation of the departure/arrival
intrinsic delay distributions, avoiding a degenerate zero-variance
LogNormal/Normal distribution.
"""
const SHIFT_TO_AVOID_ZERO = 1.0e-3
