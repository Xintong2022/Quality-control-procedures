"""
travel_time.py — the ONE canonical door-to-door travel-time model for TREK.

Imported by BOTH the scorer's B3 (spatio-temporal feasibility) and the agent-facing
`compute_travel_time` API tool, so the minimum gap the tool reports to the agent is
EXACTLY the gap B3 will require between two consecutive same-day events. Any drift here
would score the agent on a feasibility rule it was told something different about, which
is the unfairness this module exists to remove.
"""

import math

GROUND_SPEED_KMH = 60      # door-to-door surface average, incl. waiting / parking / walking
GROUND_BUFFER_MIN = 15     # fixed surface overhead (getting out, parking, walking the last leg)
MIN_TRANSFER_MIN = 45      # floor for any hop, however short
AIR_OVERHEAD_MIN = 180     # end-to-end airport overhead: transfer, check-in, security, baggage
AIR_SPEED_KMH = 700        # cruise speed once airborne

# Kept for backwards compatibility with callers that imported these names.
FLIGHT_BUFFER_MIN = AIR_OVERHEAD_MIN
FLIGHT_DISTANCE_KM = 300


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _ground_minutes(distance_km):
    return max(MIN_TRANSFER_MIN, GROUND_BUFFER_MIN + distance_km / GROUND_SPEED_KMH * 60.0)


def _air_minutes(distance_km):
    return AIR_OVERHEAD_MIN + distance_km / AIR_SPEED_KMH * 60.0


def min_travel_minutes(distance_km):
    """Minimum minutes required to cover `distance_km` door to door, by the faster of the two modes.

    minutes(d) = min( max(45, 15 + d/60*60),          # surface
                      180 + d/700*60 )                # air, incl. 3 h airport overhead

    Taking the MINIMUM over modes makes the requirement monotone non-decreasing in distance. The
    previous form returned a flat 90 min for anything beyond 300 km while charging 465 min at
    exactly 300 km — a 5.2x DROP as the trip got longer, which is indefensible and let a
    distance-blind constant-gap schedule pass. Surface wins below ~180 km, air above it.
    """
    d = max(0.0, float(distance_km))
    return int(round(min(_ground_minutes(d), _air_minutes(d))))


def travel_mode(distance_km):
    """Which mode the minimum is achieved by — reported to the agent so the two agree."""
    d = max(0.0, float(distance_km))
    return "ground" if _ground_minutes(d) <= _air_minutes(d) else "flight"
