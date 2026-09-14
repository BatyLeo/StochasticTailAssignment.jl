"""
$TYPEDSIGNATURES

Remove all columns that are incompatible with the given route being in the the solution.
"""
function filter_proper_columns!(mask, columns, route)
    i = route.immat_index

    for (column_index, column) in enumerate(columns)
        mask[column_index] || continue  # Skip already filtered columns

        # Remove all columns with the same immat, and all columns with common activities
        if column.immat_index == i || !isdisjoint(column.route, route.route)
            mask[column_index] = false
            continue
        end
    end

    return nothing
end

function filter_proper_columns!(columns::Vector{Route}, route::Route)
    i = route.immat_index

    filter!(columns) do column
        # Remove all columns with the same immat, and all columns with common activities
        return immat_index(column) != i && isdisjoint(column.route, route.route)
    end

    return nothing
end

"""
$TYPEDSIGNATURES

Apply a (pure) diving heuristic to the instance, using the given columns and scenarios.

# Arguments
- `instance::Instance`: the instance to solve
- `columns`: output of the column generation
- `root_delays`: root delay scenarios
- `dual_values`: dual variable values at the end of the column generation
"""
function pure_diving_heuristic(
    instance::ActivitySchedule,
    columns,
    root_delays,
    dual_values;
    model_builder,
    delay_cost_function,
    silent=false,
)
    best_column_index = argmax(dual_values)
    best_column = columns[best_column_index]
    partial_solution = [best_column]

    masked_instance = MaskedSchedule(instance, best_column)

    initial_paths = deepcopy(columns)
    filter_proper_columns!(initial_paths, best_column)

    it = 0
    while !is_feasible(partial_solution, instance; verbose=false)
        it += 1
        silent || @info "Diving heuristic iteration $it: $best_column"

        res = stochastic_column_generation(
            masked_instance,
            initial_paths;
            model_builder,
            root_delays,
            delay_cost_function,
            max_nb_columns=40000,
            tol=1e-4,
            silent=true,
            starting_sweep=true,
        )

        if !res.feasible
            error("Column generation infeasible during diving heuristic iteration $it")
        end

        best_column_index = argmax(res.dual_values)
        best_column = res.columns[best_column_index]
        push!(partial_solution, best_column)

        mask_schedule!(masked_instance, best_column)
        initial_paths = deepcopy(res.columns)
        filter_proper_columns!(initial_paths, best_column)
    end

    return partial_solution
end

function find_best_non_taboo_column(columns, dual_values, taboo_list)
    order = sortperm(dual_values; rev=true)
    for i in order
        best_column = columns[i]
        if !(best_column in taboo_list)
            return best_column
        end
    end
    return nothing
end

function diving_heuristic_with_backtracking!(
    instance::AbstractSchedule,
    columns,
    root_delays,
    dual_values,
    max_discrepancy=1,
    taboo_list=Route[],
    partial_solution=Route[];
    model_builder,
    delay_cost_function,
    depth=1,
    silent=true,
    tol=1e-6,
)
    current_solution = deepcopy(partial_solution)

    while length(taboo_list) <= max_discrepancy
        best_column = find_best_non_taboo_column(columns, dual_values, taboo_list)
        silent || @info "$(length(taboo_list)) | $depth"
        if isnothing(best_column)
            return current_solution, false
        end
        current_solution = vcat(partial_solution, best_column)
        if is_feasible(current_solution, instance; verbose=false)
            return current_solution, true
        end

        masked_instance = MaskedSchedule(instance, best_column)
        initial_paths = deepcopy(columns)
        filter_proper_columns!(initial_paths, best_column)

        res = try
            stochastic_column_generation(
                masked_instance,
                initial_paths;
                model_builder,
                root_delays,
                delay_cost_function,
                max_nb_columns=40000,
                tol,
                silent=true,
                starting_sweep=true,
            )
        catch
            # Current instance is infeasible
            (; feasible=false)
        end

        if !res.feasible
            push!(taboo_list, best_column)
            # @info "Route makes infeasible: $best_column | $(length(taboo_list)) | $depth"
            continue
        end

        solution, feasible = diving_heuristic_with_backtracking!(
            masked_instance,
            res.columns,
            root_delays,
            res.dual_values,
            max_discrepancy,
            deepcopy(taboo_list),
            current_solution;
            model_builder,
            delay_cost_function,
            depth=depth + 1,
            silent,
        )

        if feasible
            return solution, true
        end

        # else continue
        push!(taboo_list, best_column)
        # @info "B $taboo_list"
    end

    return current_solution, false
end
