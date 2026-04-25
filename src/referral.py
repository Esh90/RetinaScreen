import requests
import folium
from streamlit_folium import st_folium
import streamlit as st


def find_nearest_ophthalmologists(city: str, radius_km: int = 25) -> list:
    """Geocode city → Overpass API query. No API key needed."""
    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={'q': city, 'format': 'json', 'limit': 1},
            headers={'User-Agent': 'RetinaScreen/1.0'},
            timeout=8
        ).json()
        if not geo:
            return []
        lat, lon = float(geo[0]['lat']), float(geo[0]['lon'])
    except Exception:
        return []

    r = radius_km * 1000
    query = f"""
    [out:json][timeout:25];
    (
      node["healthcare"="doctor"]["healthcare:speciality"="ophthalmology"](around:{r},{lat},{lon});
      node["amenity"="clinic"]["healthcare:speciality"="ophthalmology"](around:{r},{lat},{lon});
      node["amenity"="hospital"](around:{r},{lat},{lon});
    );
    out body;
    """
    try:
        elements = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query, timeout=20
        ).json().get('elements', [])
    except Exception:
        return []

    results = []
    for el in elements[:12]:
        tags = el.get('tags', {})
        results.append({
            'name':    tags.get('name', 'Hospital / Clinic'),
            'lat':     el.get('lat', lat),
            'lon':     el.get('lon', lon),
            'address': tags.get('addr:street', tags.get('addr:full', 'See map')),
            'phone':   tags.get('phone', tags.get('contact:phone', 'N/A')),
            'hours':   tags.get('opening_hours', 'N/A'),
        })
    return results


def render_specialist_map(specialists: list) -> None:
    if not specialists:
        st.warning("No specialists found. Try increasing the search radius.")
        return

    cx, cy = specialists[0]['lat'], specialists[0]['lon']
    m = folium.Map(location=[cx, cy], zoom_start=13, tiles='CartoDB dark_matter')

    for i, sp in enumerate(specialists):
        folium.Marker(
            location=[sp['lat'], sp['lon']],
            popup=folium.Popup(
                f"<b>{sp['name']}</b><br>{sp['address']}<br>📞 {sp['phone']}",
                max_width=260
            ),
            tooltip=sp['name'],
            icon=folium.Icon(color='red', icon='plus-sign', prefix='glyphicon')
        ).add_to(m)

    st_folium(m, width=None, height=380, returned_objects=[])