"""
referral.py — Ophthalmology referral search via Nominatim + Overpass API.
Robust error handling, haversine distance, Google Maps directions (lat, lon).
"""

from __future__ import annotations

import logging
import math
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "RetinaScreen/3.0 (clinical-research; contact: retina-screen-local)"
GEOCODE_TIMEOUT = 12
OVERPASS_TIMEOUT = 45

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def google_maps_directions_url(lat: float, lon: float) -> str:
    """Driving directions deep link (destination as coordinates)."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.7f},{lon:.7f}"


def _geocode_city(city: str) -> Tuple[Optional[float], Optional[float]]:
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
    except requests.RequestException as e:
        logger.warning("Nominatim geocode failed: %s", e)
        return None, None
    except (ValueError, KeyError, IndexError, TypeError) as e:
        logger.warning("Nominatim parse error: %s", e)
        return None, None


def _overpass_query(lat: float, lon: float, radius_m: int) -> List[dict]:
    """
    Facilities: ophthalmology doctors (healthcare=doctor + speciality) and hospitals.
    """
    query = f"""
    [out:json][timeout:60];
    (
      node["healthcare"="doctor"]["healthcare:speciality"="ophthalmology"](around:{radius_m},{lat},{lon});
      way["healthcare"="doctor"]["healthcare:speciality"="ophthalmology"](around:{radius_m},{lat},{lon});
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center body;
    """
    last_err: Optional[Exception] = None
    for url in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(
                url,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=OVERPASS_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except requests.RequestException as e:
            last_err = e
            logger.warning("Overpass request failed (%s): %s", url, e)
            continue
    if last_err:
        logger.error("All Overpass endpoints failed: %s", last_err)
    return []


def _element_coords(el: dict, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    if el.get("type") == "way" and "center" in el:
        return float(el["center"]["lat"]), float(el["center"]["lon"])
    lat, lon = el.get("lat"), el.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    return origin_lat, origin_lon


def _format_address(tags: dict, city_fallback: str) -> str:
    parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", tags.get("addr:suburb", "")),
        tags.get("addr:postcode", ""),
    ]
    line = ", ".join(p.strip() for p in parts if p and str(p).strip())
    if line:
        return line
    if tags.get("addr:full"):
        return str(tags["addr:full"]).strip()
    return city_fallback


def _phone_from_tags(tags: dict) -> str:
    for key in ("phone", "contact:phone", "contact:mobile"):
        v = tags.get(key)
        if v:
            return str(v).strip()
    return "Not listed"


def _parse_elements(
    elements: List[dict],
    origin_lat: float,
    origin_lon: float,
    city: str,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()

    for el in elements:
        tags = el.get("tags") or {}
        is_oph = tags.get("healthcare") == "doctor" and tags.get("healthcare:speciality") == "ophthalmology"
        is_hosp = tags.get("amenity") == "hospital"
        if not (is_oph or is_hosp):
            continue

        name = (tags.get("name") or "").strip()
        if not name:
            name = "Ophthalmology clinic" if is_oph else "Hospital"

        lat, lon = _element_coords(el, origin_lat, origin_lon)
        dedupe_key = (name.lower()[:80], round(lat, 4), round(lon, 4))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        address = _format_address(tags, city)
        phone = _phone_from_tags(tags)
        dist_km = _haversine_km(origin_lat, origin_lon, lat, lon)

        rows.append(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "address": address or "Address not available",
                "phone": phone,
                "distance_km": dist_km,
                "maps_url": google_maps_directions_url(lat, lon),
                "kind": "Ophthalmologist" if is_oph else "Hospital",
            }
        )

    rows.sort(key=lambda r: r["distance_km"])
    return rows[:max_results]


def find_nearest_ophthalmologists(city: str, radius_km: int = 30) -> List[Dict[str, Any]]:
    """
    Geocode city, query Overpass for ophthalmology doctors and hospitals, return sorted results.
    Each dict: name, lat, lon, address, phone, distance_km, maps_url, kind.
    """
    city = (city or "").strip()
    if not city:
        return []

    lat, lon = _geocode_city(city)
    if lat is None or lon is None:
        return []

    radius_m = max(1000, min(int(radius_km * 1000), 200_000))
    elements = _overpass_query(lat, lon, radius_m)
    return _parse_elements(elements, lat, lon, city)


def render_specialist_map(
    specialists: List[Dict[str, Any]],
    city: str = "",
    dark_map: bool = False,
) -> None:
    if not specialists:
        st.warning(
            "No facilities matched the query. Try a larger city, increase radius, "
            "or verify spelling — OSM coverage varies by region."
        )
        return

    lat0, lon0 = specialists[0]["lat"], specialists[0]["lon"]
    tiles = "CartoDB dark_matter" if dark_map else "CartoDB positron"
    m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles=tiles, attr="CartoDB")

    for sp in specialists:
        d = sp.get("distance_km", 0.0)
        popup_html = (
            f"<div style='font-family:system-ui,sans-serif;font-size:13px;min-width:200px;'>"
            f"<strong style='font-size:14px'>{sp['name']}</strong><br>"
            f"<span style='color:#64748b'>{sp.get('kind','')}</span><br>"
            f"<span style='color:#334155'>{sp['address']}</span><br>"
            f"<span style='color:#334155'>{sp['phone']}</span><br>"
            f"<span style='color:#0f172a;font-weight:600'>{d:.1f} km</span><br>"
            f"<a href='{sp['maps_url']}' target='_blank' rel='noopener' "
            f"style='color:#0d9488;font-weight:600'>Get directions</a>"
            f"</div>"
        )
        color = "green" if sp.get("kind") == "Ophthalmologist" else "red"
        folium.Marker(
            location=[sp["lat"], sp["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{sp['name']} ({d:.1f} km)",
            icon=folium.Icon(color=color, icon="plus-sign", prefix="glyphicon"),
        ).add_to(m)

    st_folium(m, width="100%", height=380, returned_objects=[])
