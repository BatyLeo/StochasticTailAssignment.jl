@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(
        StochasticTailAssignment; deps_compat=(check_extras=false,), ambiguities=false
    )
end

@testitem "Code linting (JET.jl)" begin
    using JET
    JET.test_package(StochasticTailAssignment; target_modules=[StochasticTailAssignment])
end

@testitem "Formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    @test format(StochasticTailAssignment; verbose=false, overwrite=false)
end
