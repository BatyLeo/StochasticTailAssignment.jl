module StochasticTailAssignmentMakieExt

using Makie
using Dates
using Statistics: mean

using StochasticTailAssignment: StochasticTailAssignment, plot_gantt
using StochasticTailAssignment.AircraftRoutingBase:
    ActivitySchedule,
    Route,
    Leg,
    Immat,
    departure_time,
    arrival_time,
    departure_airport,
    arrival_airport,
    leg_indices_from_route,
    immat_index,
    id,
    operational_cost,
    nb_immats,
    nb_legs,
    leg_indices
using StochasticTailAssignment.FlightDelayModel:
    propagate_delays_from_root_delays, delay_expected_cost

const AIRPORT_PALETTE = [
    colorant"#2196F3",
    colorant"#4CAF50",
    colorant"#FF9800",
    colorant"#9C27B0",
    colorant"#00BCD4",
    colorant"#795548",
    colorant"#E91E63",
    colorant"#607D8B",
    colorant"#8BC34A",
    colorant"#FF5722",
    colorant"#3F51B5",
    colorant"#009688",
    colorant"#CDDC39",
    colorant"#FFC107",
    colorant"#673AB7",
    colorant"#03A9F4",
    colorant"#F44336",
    colorant"#4DB6AC",
    colorant"#FFB74D",
    colorant"#CE93D8",
]

"""
Build a mapping from leg index to assigned aircraft index, from a vector of routes.
"""
function _build_assignment(routes, instance)
    asgn = Dict{Int,Int}()
    for r in routes
        for li in leg_indices_from_route(r, instance)
            asgn[li] = immat_index(r)
        end
    end
    return asgn
end

"""
Convert a `DateTime` into a number of hours elapsed since `t_start`.
"""
function _hours_from_start(t, t_start)
    return Dates.value(t - t_start) / 3_600_000
end

"""
Render one Gantt chart section (one axis) for a given leg-to-aircraft assignment.

`diff_assignment` is an optional comparison assignment, used to highlight legs whose
aircraft assignment differs between the two solutions.
"""
function _render_section!(
    fig_position,
    legs,
    immats,
    t_start,
    total_hours,
    assignment,
    section_title,
    avg_delays,
    diff_assignment,
    airport_colors,
    max_delay,
    show_delays,
)
    n_ac = length(immats)
    ax = Axis(
        fig_position;
        yticks=(1:n_ac, [im.id for im in immats]),
        xlabel="Time (hours)",
        title=section_title,
        yreversed=true,
        autolimitaspect=nothing,
    )

    dep_rects = Makie.Rect2f[]
    arr_rects = Makie.Rect2f[]
    full_rects = Makie.Rect2f[]
    dep_ap_colors = Any[]
    arr_ap_colors = Any[]
    delay_vals = Float64[]
    labels = String[]
    diff_rects = Makie.Rect2f[]

    for li in eachindex(legs)
        haskey(assignment, li) || continue
        ac = assignment[li]
        leg = legs[li]

        dep_h = _hours_from_start(departure_time(leg), t_start)
        arr_h = _hours_from_start(arrival_time(leg), t_start)
        mid_h = (dep_h + arr_h) / 2

        bar_h = 0.8
        y = ac - 0.4

        push!(dep_rects, Makie.Rect2f(dep_h, y, mid_h - dep_h, bar_h))
        push!(arr_rects, Makie.Rect2f(mid_h, y, arr_h - mid_h, bar_h))
        push!(full_rects, Makie.Rect2f(dep_h, y, arr_h - dep_h, bar_h))

        push!(dep_ap_colors, airport_colors[departure_airport(leg)])
        push!(arr_ap_colors, airport_colors[arrival_airport(leg)])

        delay_val = avg_delays === nothing ? 0.0 : Float64(avg_delays[li])
        push!(delay_vals, delay_val)

        delay_line =
            avg_delays === nothing ? "" : "\nAvg delay: $(round(delay_val; digits=1)) min"
        push!(
            labels,
            "$(id(leg))\n$(departure_airport(leg)) → $(arrival_airport(leg))\n" *
            "$(Dates.format(departure_time(leg), "HH:MM")) - $(Dates.format(arrival_time(leg), "HH:MM"))" *
            delay_line,
        )

        if diff_assignment !== nothing && get(diff_assignment, li, -1) != ac
            push!(
                diff_rects,
                Makie.Rect2f(dep_h - 0.05, y - 0.05, arr_h - dep_h + 0.1, bar_h + 0.1),
            )
        end
    end

    ap_vis = @lift !$show_delays
    poly!(ax, dep_rects; color=dep_ap_colors, visible=ap_vis, inspectable=false)
    poly!(ax, arr_rects; color=arr_ap_colors, visible=ap_vis, inspectable=false)

    if avg_delays !== nothing
        del_vis = show_delays
        poly!(
            ax,
            full_rects;
            color=delay_vals,
            colormap=Reverse(:RdYlBu),
            colorrange=(0.0, max(max_delay, 1.0)),
            visible=del_vis,
            inspectable=false,
        )
    end

    if !isempty(diff_rects)
        poly!(
            ax,
            diff_rects;
            color=(:transparent, 0.0),
            strokecolor=:orange,
            strokewidth=2,
            inspectable=false,
        )
    end

    scatter!(
        ax,
        [r.origin[1] + r.widths[1] / 2 for r in full_rects],
        [r.origin[2] + r.widths[2] / 2 for r in full_rects];
        marker=:rect,
        markersize=[Vec2f(r.widths[1], r.widths[2]) for r in full_rects],
        markerspace=:data,
        color=[(:black, 0.01) for _ in full_rects],
        inspector_label=(self, i, p) -> labels[i],
        inspectable=true,
    )

    xlims!(ax, -0.5, total_hours + 0.5)
    ylims!(ax, 0.2, n_ac + 0.8)

    return ax
end

"""
Plot an interactive Gantt chart of an aircraft routing solution.

Each row corresponds to an aircraft immatriculation, and each bar corresponds to a leg,
colored by departure and arrival airport (left and right half of the bar respectively).

# Keyword arguments

- `root_delays`: optional matrix of sampled root delays, used together with
  `delay_cost_function` to compute and display average leg delays.
- `delay_cost_function`: cost function associated with `root_delays`.
- `comparison_routes`: optional second set of routes (e.g. a deterministic solution),
  displayed in a second section below the primary one for visual comparison.
- `comparison_label`: title used for the comparison section.
- `title`: figure title, currently unused but kept for API symmetry.
"""
function StochasticTailAssignment.plot_gantt(
    instance::ActivitySchedule,
    routes::Vector{Route};
    root_delays::Union{Nothing,AbstractMatrix}=nothing,
    delay_cost_function=nothing,
    comparison_routes::Union{Nothing,Vector{Route}}=nothing,
    comparison_label::String="Stochastic",
    title::String="Aircraft Routing Solution",
    figure_kwargs...,
)
    legs = instance.legs
    immats = instance.immats
    t_start = minimum(departure_time(l) for l in legs)

    asgn = _build_assignment(routes, instance)
    has_comp = comparison_routes !== nothing
    comp_asgn = has_comp ? _build_assignment(comparison_routes, instance) : nothing

    airports = sort(
        unique(
            vcat([departure_airport(l) for l in legs], [arrival_airport(l) for l in legs])
        ),
    )
    airport_colors = Dict(
        ap => AIRPORT_PALETTE[mod1(i, length(AIRPORT_PALETTE))] for
        (i, ap) in enumerate(airports)
    )

    has_delays = root_delays !== nothing && delay_cost_function !== nothing

    avg_delay = nothing
    comp_avg_delay = nothing
    max_delay = 1.0
    if has_delays
        ad = propagate_delays_from_root_delays(routes, root_delays, instance)
        avg_delay = vec(mean(max.(ad, 0.0f0); dims=1))
        max_delay = maximum(avg_delay)
        if has_comp
            comp_ad = propagate_delays_from_root_delays(
                comparison_routes, root_delays, instance
            )
            comp_avg_delay = vec(mean(max.(comp_ad, 0.0f0); dims=1))
            max_delay = max(max_delay, maximum(comp_avg_delay))
        end
    end

    n_ac = length(immats)
    t_end = maximum(arrival_time(l) for l in legs)
    total_hours = _hours_from_start(t_end, t_start)
    chart_height = max(300, n_ac * 60)
    fig_height = chart_height + (has_comp ? chart_height : 0) + 120
    fig = Figure(; size=(1400, fig_height), figure_kwargs...)

    if has_delays
        tg = fig[1, 1] = GridLayout(; tellwidth=false)
        toggle = Toggle(tg[1, 1]; active=false)
        Label(tg[1, 2], "Show delays")
        show_delays = toggle.active
    else
        show_delays = Observable(false)
    end

    if has_comp
        _render_section!(
            fig[has_delays ? 2 : 1, 1],
            legs,
            immats,
            t_start,
            total_hours,
            asgn,
            "Deterministic",
            avg_delay,
            comp_asgn,
            airport_colors,
            max_delay,
            show_delays,
        )
        _render_section!(
            fig[has_delays ? 3 : 2, 1],
            legs,
            immats,
            t_start,
            total_hours,
            comp_asgn,
            comparison_label,
            comp_avg_delay,
            asgn,
            airport_colors,
            max_delay,
            show_delays,
        )
    else
        _render_section!(
            fig[has_delays ? 2 : 1, 1],
            legs,
            immats,
            t_start,
            total_hours,
            asgn,
            title,
            avg_delay,
            nothing,
            airport_colors,
            max_delay,
            show_delays,
        )
    end

    legend_elements = [Makie.PolyElement(; color=airport_colors[ap]) for ap in airports]
    legend_row = (has_comp ? 3 : 2) + has_delays
    Legend(
        fig[legend_row, 1],
        legend_elements,
        airports;
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        nbanks=(length(airports) > 10 ? 2 : 1),
    )
    rowgap!(fig.layout, 10)
    if has_delays
        rowsize!(fig.layout, 1, Auto(0.05))
    end
    first_chart_row = 1 + has_delays
    for r in first_chart_row:(first_chart_row + (has_comp ? 1 : 0))
        rowsize!(fig.layout, r, Auto(1.0))
    end
    rowsize!(fig.layout, legend_row, Auto(0.15))

    DataInspector(fig)

    return fig
end

end
