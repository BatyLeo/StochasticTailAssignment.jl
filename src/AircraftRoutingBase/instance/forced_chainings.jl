"""
$TYPEDEF

Data structure defining an forced chaining between two activities.

# Fields
$TYPEDFIELDS
"""
@kwdef struct ForcedChaining
    "id of the first activity"
    first_activity_id::String
    "id of the second activity chained to the first one"
    second_activity_id::String
end
