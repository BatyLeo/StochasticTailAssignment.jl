function string_to_date(s::AbstractString)
    return Dates.DateTime(s, dateformat"yyyy-mm-ddTHH:MM:SSZ")
end

to_minutes(t::Dates.Millisecond) = t.value / 1000 / 60
