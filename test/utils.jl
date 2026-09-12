@testitem "Utility functions" begin
    using Dates
    using StochasticTailAssignment: to_minutes, string_to_date
    @test to_minutes(Dates.Millisecond(60000)) == 1.0
    @test string_to_date("2022-01-01T23:34:10") == DateTime(2022, 1, 1, 23, 34, 10)
end
