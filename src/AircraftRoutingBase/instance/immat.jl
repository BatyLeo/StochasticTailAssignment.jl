"""
$TYPEDEF

Data structure defining an aircraft immatriculation.

# Fields
$TYPEDFIELDS
"""
@kwdef struct Immat
    "unique identifier string"
    id::String
    "fuel factor, the percentage of extra fuel consumption for the aircraft type"
    fuel_factor::Float64
    "id of the last activity operated by this immatriculation"
    last_activity_id::String
    "aircraft type code of the immatriculation"
    aircraft_type::String
end

# Base.:(==)(a::Immat, b::Immat) = a.id == b.id
aircraft_type(immat::Immat) = immat.aircraft_type
