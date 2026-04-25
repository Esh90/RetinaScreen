"""
referral.py — Specialist Finder Module
Uses OpenStreetMap Nominatim for geocoding and Overpass API for facility lookup.
No API keys required. All services are free and open.
"""

import requests
import urllib.parse
import folium
from streamlit_folium import st_folium
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL     = "https://overpass-api.de/api/interpreter"
USER_AGENT       = "RetinaScreen/2.0 (clinical-ai-tool)"
GEOCODE_TIMEOUT  = 8   # seconds
OVERPASS_TIMEOUT = 22  # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_nearest_ophthalmologists(city: str, radius_km: int = 25) -> list:
    """
    Geocode `city` via Nominatim, then query Overpass for ophthalmic facilities.
    Returns a list of dicts with name, lat, lon, address, phone, hours, maps_url.
    Returns [] on any failure.
    """
    lat, lon = _geocode_city(city)
    if lat is None:
        return []

    elements = _overpass_query(lat, lon, radius_km * 1000)
    return _parse_elements(elements, lat, lon, city)


def render_specialist_map(specialists: list, city: str = "") -> None:
    """
    Render an interactive Folium map embedded in the Streamlit app.
    Falls back gracefully if the list is empty.
    """
    if not specialists:
        st.warning("No facilities found. Try increasing the search radius or entering a larger nearby city.")
        return

    cx, cy = specialists[0]["lat"], specialists[0]["lon"]

    # Choose tile layer based on Streamlit's effective theme
    m = folium.Map(
        location=[cx, cy],
        zoom_start=13,
        tiles="CartoDB positron",   # clean, neutral light tile
        attr="CartoDB"
    )

    for sp in specialists:
        popup_html = (
            f"<div style='font-family: system-ui; font-size: 13px; min-width: 180px;'>"
            f"<strong style='font-size:14px'>{sp['name']}</strong><br>"
            f"<span style='color:#555'>{sp['address']}</span><br>"
            f"<span style='color:#555'>{sp['phone']}</span><br>"
            f"<a href='{sp['maps_url']}' target='_blank' "
            f"style='color:#0D7377;font-weight:600'>Get Directions</a>"
            f"</div>"
        )
        folium.Marker(
            location=[sp["lat"], sp["lon"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=sp["name"],
            icon=folium.Icon(color="darkblue", icon="plus-sign", prefix="glyphicon"),
        ).add_to(m)

    st_folium(m, width="100%", height=360, returned_objects=[])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _geocode_city(city: str):
    """Return (lat, lon) floats or (None, None) on failure."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=GEOCODE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None, None


def _overpass_query(lat: float, lon: float, radius_m: int) -> list:
    """
    Query Overpass for hospitals, clinics, and optometrists.
    Uses a broad query so results are useful even in regions with sparse tagging.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["healthcare"="doctor"]["healthcare:speciality"="ophthalmology"](around:{radius_m},{lat},{lon});
      way["healthcare"="doctor"]["healthcare:speciality"="ophthalmology"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"]["healthcare:speciality"="ophthalmology"](around:{radius_m},{lat},{lon});
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["healthcare"="optometrist"](around:{radius_m},{lat},{lon});
      node["healthcare"="ophthalmologist"](around:{radius_m},{lat},{lon});
    );
    out center body;
    """
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=OVERPASS_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("elements", [])
    except Exception:
        return []


def _parse_elements(elements: list, fallback_lat: float, fallback_lon: float,
                    city: str) -> list:
    """Normalise Overpass elements into a clean list of dicts."""
    results = []
    seen_names = set()

    for el in elements[:15]:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            name = "Hospital / Clinic"

        # De-duplicate by name
        if name in seen_names:
            continue
        seen_names.add(name)

        # Coordinates: nodes give lat/lon directly; ways give centre
        if el.get("type") == "way" and "center" in el:
            lat = el["center"]["lat"]
            lon = el["center"]["lon"]
        else:
            lat = el.get("lat", fallback_lat)
            lon = el.get("lon", fallback_lon)

        # Address assembly — prefer structured tags, fall back to full-address or city
        parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ]
        address = ", ".join(p for p in parts if p).strip(", ") or tags.get("addr:full", city)

        # Google Maps deep-link for routing
        query_str = urllib.parse.quote(f"{name} {address or city}")
        maps_url = f"https://www.google.com/maps/search/?api=1&query={query_str}"

        results.append({
            "name":     name,
            "lat":      lat,
            "lon":      lon,
            "address":  address or "Address not available",
            "phone":    tags.get("phone", tags.get("contact:phone", "Not listed")),
            "hours":    tags.get("opening_hours", "Not listed"),
            "maps_url": maps_url,
        })

    return results