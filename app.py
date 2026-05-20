import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go

from geopy.geocoders import Nominatim
from skyfield.api import load, Star, wgs84
from skyfield.data import hipparcos


# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Celestial Sky Renderer",
    page_icon="🔭",
    layout="wide"
)

EPHEMERIS_FILE = "de440.bsp"


# ==========================================================
# COMMON STAR NAME MAP — HIP ID → NAME
# ==========================================================

HIP_NAMES = {
    32349: "Sirius",
    30438: "Canopus",
    69673: "Arcturus",
    71683: "Alpha Centauri",
    91262: "Vega",
    24608: "Capella",
    24436: "Rigel",
    37279: "Procyon",
    27989: "Betelgeuse",
    7588: "Achernar",
    97649: "Altair",
    60718: "Acrux",
    21421: "Aldebaran",
    25336: "Bellatrix",
    26311: "Alnilam",
    26727: "Alnitak",
    25930: "Mintaka",
    27366: "Saiph",
    65474: "Spica",
    80763: "Antares",
    102098: "Deneb",
    113368: "Fomalhaut",
    54061: "Dubhe",
    53910: "Merak",
    58001: "Phecda",
    59774: "Megrez",
    62956: "Alioth",
    65378: "Mizar",
    67301: "Alkaid",
    11767: "Polaris",
    68702: "Hadar",
    62434: "Mimosa",
    68702: "Beta Centauri",
    37826: "Pollux",
    36850: "Castor",
    49669: "Regulus",
    85927: "Rasalhague",
    87833: "Kaus Australis",
    90185: "Nunki",
    92855: "Albireo",
}


# ==========================================================
# CONSTELLATION FALLBACK LINES — HIP ID PAIRS
# ==========================================================

CONSTELLATION_LINES = {
    "Orion": [
        (27989, 25336),
        (25336, 25930),
        (25930, 26311),
        (26311, 26727),
        (26727, 27366),
        (27366, 24436),
        (27989, 24436),
        (25336, 27366),
    ],
    "Ursa Major": [
        (54061, 53910),
        (53910, 58001),
        (58001, 59774),
        (59774, 62956),
        (62956, 65378),
        (65378, 67301),
    ],
    "Summer Triangle": [
        (91262, 102098),
        (102098, 97649),
        (97649, 91262),
    ],
    "Canis Major / Minor": [
        (32349, 37279),
    ],
}


# ==========================================================
# MESSIER DSO CATALOG SAMPLE
# RA in degrees, Dec in degrees, mag approximate
# You can expand this table later.
# ==========================================================

MESSIER_OBJECTS = [
    {"name": "M1 Crab Nebula", "type": "Supernova Remnant", "ra": 83.633, "dec": 22.014, "mag": 8.4},
    {"name": "M8 Lagoon Nebula", "type": "Nebula", "ra": 270.925, "dec": -24.380, "mag": 6.0},
    {"name": "M13 Hercules Globular Cluster", "type": "Globular Cluster", "ra": 250.423, "dec": 36.461, "mag": 5.8},
    {"name": "M16 Eagle Nebula", "type": "Nebula", "ra": 274.700, "dec": -13.806, "mag": 6.0},
    {"name": "M17 Omega Nebula", "type": "Nebula", "ra": 275.196, "dec": -16.171, "mag": 6.0},
    {"name": "M20 Trifid Nebula", "type": "Nebula", "ra": 270.650, "dec": -23.030, "mag": 6.3},
    {"name": "M27 Dumbbell Nebula", "type": "Planetary Nebula", "ra": 299.902, "dec": 22.721, "mag": 7.5},
    {"name": "M31 Andromeda Galaxy", "type": "Galaxy", "ra": 10.685, "dec": 41.269, "mag": 3.4},
    {"name": "M33 Triangulum Galaxy", "type": "Galaxy", "ra": 23.462, "dec": 30.660, "mag": 5.7},
    {"name": "M42 Orion Nebula", "type": "Nebula", "ra": 83.822, "dec": -5.391, "mag": 4.0},
    {"name": "M44 Beehive Cluster", "type": "Open Cluster", "ra": 130.100, "dec": 19.667, "mag": 3.7},
    {"name": "M45 Pleiades", "type": "Open Cluster", "ra": 56.750, "dec": 24.117, "mag": 1.6},
    {"name": "M51 Whirlpool Galaxy", "type": "Galaxy", "ra": 202.470, "dec": 47.195, "mag": 8.4},
    {"name": "M57 Ring Nebula", "type": "Planetary Nebula", "ra": 283.396, "dec": 33.030, "mag": 8.8},
    {"name": "M81 Bode's Galaxy", "type": "Galaxy", "ra": 148.888, "dec": 69.066, "mag": 6.9},
    {"name": "M82 Cigar Galaxy", "type": "Galaxy", "ra": 148.969, "dec": 69.679, "mag": 8.4},
    {"name": "M87 Virgo A", "type": "Galaxy", "ra": 187.706, "dec": 12.391, "mag": 8.6},
    {"name": "M101 Pinwheel Galaxy", "type": "Galaxy", "ra": 210.803, "dec": 54.349, "mag": 7.9},
    {"name": "M104 Sombrero Galaxy", "type": "Galaxy", "ra": 189.998, "dec": -11.623, "mag": 8.0},
]


# ==========================================================
# LOADERS
# ==========================================================

@st.cache_resource
def load_ephemeris():
    return load(EPHEMERIS_FILE)


@st.cache_resource
def load_timescale():
    return load.timescale()


@st.cache_data
def load_hipparcos_catalog():
    with load.open(hipparcos.URL) as f:
        stars = hipparcos.load_dataframe(f)

    stars = stars.dropna(subset=["ra_degrees", "dec_degrees", "magnitude"])
    return stars


@st.cache_data
def geocode_city(city_name: str):
    geolocator = Nominatim(user_agent="celestial_sky_renderer")
    location = geolocator.geocode(city_name, timeout=10)

    if location is None:
        return None

    return {
        "lat": float(location.latitude),
        "lon": float(location.longitude),
        "display_name": location.address,
    }


# ==========================================================
# ASTRONOMY HELPERS
# ==========================================================

def angular_separation_altaz(alt1, az1, alt2, az2):
    """
    Angular separation between two points on sky using alt/az in degrees.
    """
    alt1 = np.radians(alt1)
    az1 = np.radians(az1)
    alt2 = np.radians(alt2)
    az2 = np.radians(az2)

    cos_sep = (
        np.sin(alt1) * np.sin(alt2)
        + np.cos(alt1) * np.cos(alt2) * np.cos(az1 - az2)
    )

    cos_sep = np.clip(cos_sep, -1, 1)
    return np.degrees(np.arccos(cos_sep))


def magnitude_to_size(mag, min_size=3, max_size=14):
    """
    Brighter objects have lower magnitude.
    """
    size = max_size - (mag + 1.5) * 2.2
    return float(np.clip(size, min_size, max_size))


def moon_phase_fraction(eph, observer, t):
    """
    Approximate illuminated fraction of the Moon.
    """
    sun = eph["sun"]
    moon = eph["moon"]

    e = observer.at(t)
    sun_vec = e.observe(sun).apparent()
    moon_vec = e.observe(moon).apparent()

    separation = sun_vec.separation_from(moon_vec).degrees
    phase = (1 - math.cos(math.radians(separation))) / 2

    return phase, separation


def sky_darkness_factor(sun_alt):
    """
    Rough darkness score based on Sun altitude.
    """
    if sun_alt < -18:
        return 1.0
    if sun_alt < -12:
        return 0.75
    if sun_alt < -6:
        return 0.45
    if sun_alt < 0:
        return 0.20
    return 0.05


def observation_score(alt, mag, moon_sep, sun_alt, bortle):
    """
    Simple observing score out of 100.

    Factors:
    - altitude higher is better
    - lower magnitude is better
    - farther from Moon is better
    - darker sky is better
    - lower Bortle is better
    """
    altitude_score = np.clip(alt / 80, 0, 1)
    magnitude_score = np.clip((10 - mag) / 10, 0, 1)
    moon_score = np.clip(moon_sep / 90, 0, 1)
    darkness_score = sky_darkness_factor(sun_alt)
    bortle_score = np.clip((10 - bortle) / 9, 0, 1)

    score = (
        0.35 * altitude_score
        + 0.25 * magnitude_score
        + 0.15 * moon_score
        + 0.15 * darkness_score
        + 0.10 * bortle_score
    )

    return round(float(score * 100), 1)


# ==========================================================
# OBJECT COMPUTATION
# ==========================================================

def compute_solar_system_objects(eph, observer, t, moon_alt, moon_az, sun_alt, bortle):
    bodies = {
        "Sun": eph["sun"],
        "Moon": eph["moon"],
        "Mercury": eph["mercury"],
        "Venus": eph["venus"],
        "Mars": eph["mars barycenter"],
        "Jupiter": eph["jupiter barycenter"],
        "Saturn": eph["saturn barycenter"],
        "Uranus": eph["uranus barycenter"],
        "Neptune": eph["neptune barycenter"],
    }

    rows = []

    at_time = observer.at(t)

    for name, body in bodies.items():
        apparent = at_time.observe(body).apparent()
        alt, az, distance = apparent.altaz()

        if alt.degrees > 0:
            moon_sep = angular_separation_altaz(
                alt.degrees, az.degrees, moon_alt, moon_az
            )

            rows.append({
                "name": name,
                "category": "Solar System",
                "type": "Planet/Moon/Sun",
                "alt": alt.degrees,
                "az": az.degrees,
                "mag": np.nan,
                "size": 12 if name not in ["Sun", "Moon"] else 18,
                "moon_sep": moon_sep,
                "score": observation_score(
                    alt.degrees,
                    0 if name in ["Venus", "Jupiter"] else 3,
                    moon_sep,
                    sun_alt,
                    bortle,
                ),
            })

    return pd.DataFrame(rows)


def compute_stars(stars_df, observer, t, mag_limit, moon_alt, moon_az, sun_alt, bortle):
    visible_df = stars_df[stars_df["magnitude"] <= mag_limit].copy()

    if visible_df.empty:
        return pd.DataFrame()

    star_vectors = Star.from_dataframe(visible_df)
    apparent = observer.at(t).observe(star_vectors).apparent()
    alt, az, _ = apparent.altaz()

    visible_df["alt"] = alt.degrees
    visible_df["az"] = az.degrees

    visible_df = visible_df[visible_df["alt"] > 0].copy()

    rows = []

    for hip_id, row in visible_df.iterrows():
        name = HIP_NAMES.get(int(hip_id), f"HIP {int(hip_id)}")
        moon_sep = angular_separation_altaz(
            row["alt"], row["az"], moon_alt, moon_az
        )

        rows.append({
            "name": name,
            "category": "Star",
            "type": "Star",
            "alt": float(row["alt"]),
            "az": float(row["az"]),
            "mag": float(row["magnitude"]),
            "size": magnitude_to_size(row["magnitude"]),
            "moon_sep": moon_sep,
            "score": observation_score(
                row["alt"],
                row["magnitude"],
                moon_sep,
                sun_alt,
                bortle,
            ),
        })

    return pd.DataFrame(rows)


def compute_dso(observer, t, moon_alt, moon_az, sun_alt, bortle, max_mag):
    rows = []

    for obj in MESSIER_OBJECTS:
        if obj["mag"] > max_mag:
            continue

        target = Star(
            ra_hours=obj["ra"] / 15,
            dec_degrees=obj["dec"]
        )

        apparent = observer.at(t).observe(target).apparent()
        alt, az, _ = apparent.altaz()

        if alt.degrees > 0:
            moon_sep = angular_separation_altaz(
                alt.degrees,
                az.degrees,
                moon_alt,
                moon_az
            )

            rows.append({
                "name": obj["name"],
                "category": "Deep Sky Object",
                "type": obj["type"],
                "alt": alt.degrees,
                "az": az.degrees,
                "mag": obj["mag"],
                "size": magnitude_to_size(obj["mag"], min_size=5, max_size=12),
                "moon_sep": moon_sep,
                "score": observation_score(
                    alt.degrees,
                    obj["mag"],
                    moon_sep,
                    sun_alt,
                    bortle,
                ),
            })

    return pd.DataFrame(rows)


def compute_constellation_lines(stars_df, observer, t):
    line_traces = []

    for const_name, pairs in CONSTELLATION_LINES.items():
        for hip_a, hip_b in pairs:
            if hip_a not in stars_df.index or hip_b not in stars_df.index:
                continue

            row_a = stars_df.loc[hip_a]
            row_b = stars_df.loc[hip_b]

            star_a = Star.from_dataframe(row_a)
            star_b = Star.from_dataframe(row_b)

            alt_a, az_a, _ = observer.at(t).observe(star_a).apparent().altaz()
            alt_b, az_b, _ = observer.at(t).observe(star_b).apparent().altaz()

            if alt_a.degrees > 0 and alt_b.degrees > 0:
                line_traces.append({
                    "constellation": const_name,
                    "r": [90 - alt_a.degrees, 90 - alt_b.degrees],
                    "theta": [az_a.degrees, az_b.degrees],
                })

    return line_traces


# ==========================================================
# PLOT
# ==========================================================

def add_object_trace(fig, df, name, color, symbol):
    if df.empty:
        return

    fig.add_trace(
        go.Scatterpolar(
            r=90 - df["alt"],
            theta=df["az"],
            mode="markers",
            name=name,
            marker=dict(
                size=df["size"],
                color=color,
                symbol=symbol,
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
            text=[
                f"<b>{row['name']}</b><br>"
                f"Type: {row['type']}<br>"
                f"Alt: {row['alt']:.1f}°<br>"
                f"Az: {row['az']:.1f}°<br>"
                f"Mag: {row['mag'] if not pd.isna(row['mag']) else 'N/A'}<br>"
                f"Moon Sep: {row['moon_sep']:.1f}°<br>"
                f"Score: {row['score']}/100"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
        )
    )


def build_sky_plot(planets_df, stars_df, dso_df, constellation_lines, title):
    fig = go.Figure()

    for line in constellation_lines:
        fig.add_trace(
            go.Scatterpolar(
                r=line["r"],
                theta=line["theta"],
                mode="lines",
                name=line["constellation"],
                line=dict(color="rgba(120,120,120,0.5)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    add_object_trace(fig, stars_df, "Stars", "white", "circle")
    add_object_trace(fig, planets_df, "Solar System", "orange", "diamond")
    add_object_trace(fig, dso_df, "Deep Sky Objects", "cyan", "star")

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=760,
        paper_bgcolor="black",
        plot_bgcolor="black",
        polar=dict(
            bgcolor="black",
            radialaxis=dict(
                range=[0, 90],
                tickvals=[0, 30, 60, 90],
                ticktext=["Zenith", "60°", "30°", "Horizon"],
                showline=False,
                gridcolor="rgba(255,255,255,0.18)",
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                gridcolor="rgba(255,255,255,0.18)",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


# ==========================================================
# SIDEBAR UI
# ==========================================================

st.title("🔭 Celestial Sky Renderer")
st.caption("Interactive scientific sky map using Skyfield, Hipparcos, Messier DSOs, Moon phase, twilight, and observing scores.")

with st.sidebar:
    st.header("Observer")

    location_mode = st.radio(
        "Location method",
        ["City name", "Manual coordinates"]
    )

    if location_mode == "City name":
        city = st.text_input("City", value="Mumbai")
        geo = geocode_city(city)

        if geo is None:
            st.error("City not found. Try manual coordinates.")
            st.stop()

        latitude = geo["lat"]
        longitude = geo["lon"]

        st.success(f"{geo['display_name']}")
        st.write(f"Lat: `{latitude:.4f}`, Lon: `{longitude:.4f}`")

    else:
        latitude = st.number_input("Latitude", value=19.0760, format="%.6f")
        longitude = st.number_input("Longitude", value=72.8777, format="%.6f")

    st.header("Time")

    timezone_name = st.selectbox(
        "Timezone",
        pytz.all_timezones,
        index=pytz.all_timezones.index("Asia/Kolkata")
    )

    tz = pytz.timezone(timezone_name)

    selected_date = st.date_input(
        "Date",
        value=datetime.now(tz).date()
    )

    selected_time = st.time_input(
        "Base time",
        value=datetime.now(tz).time().replace(microsecond=0)
    )

    hour_offset = st.slider(
        "Time slider: hours from selected time",
        min_value=-12,
        max_value=12,
        value=0,
        step=1
    )

    local_dt = tz.localize(datetime.combine(selected_date, selected_time))
    local_dt = local_dt + timedelta(hours=hour_offset)
    utc_dt = local_dt.astimezone(pytz.utc)

    st.write("Local time:", local_dt.strftime("%Y-%m-%d %H:%M:%S %Z"))
    st.write("UTC time:", utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC"))

    st.header("Visibility Filters")

    bortle = st.slider(
        "Bortle light pollution class",
        min_value=1,
        max_value=9,
        value=6,
        help="1 = pristine dark sky, 9 = inner-city sky"
    )

    star_mag_limit = st.slider(
        "Star magnitude limit",
        min_value=0.0,
        max_value=7.0,
        value=3.5,
        step=0.5
    )

    dso_mag_limit = st.slider(
        "DSO magnitude limit",
        min_value=3.0,
        max_value=12.0,
        value=9.0,
        step=0.5
    )

    show_planets = st.checkbox("Show planets/Sun/Moon", value=True)
    show_stars = st.checkbox("Show stars", value=True)
    show_constellations = st.checkbox("Show constellation lines", value=True)
    show_dso = st.checkbox("Show deep sky objects", value=True)


# ==========================================================
# COMPUTE SKY
# ==========================================================

with st.spinner("Computing sky positions..."):
    eph = load_ephemeris()
    ts = load_timescale()
    stars_catalog = load_hipparcos_catalog()

    t = ts.utc(
        utc_dt.year,
        utc_dt.month,
        utc_dt.day,
        utc_dt.hour,
        utc_dt.minute,
        utc_dt.second
    )

    observer = eph["earth"] + wgs84.latlon(
        latitude_degrees=latitude,
        longitude_degrees=longitude
    )

    at_time = observer.at(t)

    sun_alt, sun_az, _ = at_time.observe(eph["sun"]).apparent().altaz()
    moon_alt, moon_az, _ = at_time.observe(eph["moon"]).apparent().altaz()

    moon_fraction, moon_sun_sep = moon_phase_fraction(eph, observer, t)

    planets_df = compute_solar_system_objects(
        eph, observer, t,
        moon_alt.degrees, moon_az.degrees,
        sun_alt.degrees,
        bortle
    ) if show_planets else pd.DataFrame()

    stars_df = compute_stars(
        stars_catalog,
        observer,
        t,
        star_mag_limit,
        moon_alt.degrees,
        moon_az.degrees,
        sun_alt.degrees,
        bortle
    ) if show_stars else pd.DataFrame()

    dso_df = compute_dso(
        observer,
        t,
        moon_alt.degrees,
        moon_az.degrees,
        sun_alt.degrees,
        bortle,
        dso_mag_limit
    ) if show_dso else pd.DataFrame()

    constellation_lines = compute_constellation_lines(
        stars_catalog,
        observer,
        t
    ) if show_constellations else []


# ==========================================================
# SKY CONDITIONS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sun altitude", f"{sun_alt.degrees:.1f}°")

with col2:
    st.metric("Moon altitude", f"{moon_alt.degrees:.1f}°")

with col3:
    st.metric("Moon illumination", f"{moon_fraction * 100:.1f}%")

with col4:
    darkness = sky_darkness_factor(sun_alt.degrees)
    st.metric("Darkness factor", f"{darkness:.2f}")

if sun_alt.degrees < -18:
    st.success("Astronomical night: excellent darkness for deep-sky observing.")
elif sun_alt.degrees < -12:
    st.info("Nautical twilight: decent sky, but not fully dark.")
elif sun_alt.degrees < -6:
    st.warning("Civil/nautical twilight: bright sky, DSOs will be difficult.")
elif sun_alt.degrees < 0:
    st.warning("Sun is below horizon but sky is still bright.")
else:
    st.error("Sun is above the horizon. Daytime sky rendering is active.")


# ==========================================================
# RENDER SKY MAP
# ==========================================================

title = (
    f"Sky Map — Lat {latitude:.2f}, Lon {longitude:.2f} | "
    f"{local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
)

fig = build_sky_plot(
    planets_df,
    stars_df,
    dso_df,
    constellation_lines,
    title
)

st.plotly_chart(fig, width="stretch")


# ==========================================================
# OBSERVING RECOMMENDATIONS
# ==========================================================

st.subheader("Best Objects to Observe")

combined = pd.concat(
    [planets_df, stars_df, dso_df],
    ignore_index=True
)

if combined.empty:
    st.warning("No visible objects found with the current filters.")
else:
    combined = combined.sort_values("score", ascending=False)

    top_objects = combined[
        ["name", "category", "type", "alt", "az", "mag", "moon_sep", "score"]
    ].head(20).copy()

    top_objects["alt"] = top_objects["alt"].round(2)
    top_objects["az"] = top_objects["az"].round(2)
    top_objects["moon_sep"] = top_objects["moon_sep"].round(2)

    st.dataframe(
        top_objects,
        width="stretch",
        hide_index=True
    )


# ==========================================================
# EXPORT
# ==========================================================

st.subheader("Export Visible Objects")

if not combined.empty:
    csv = combined.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download visible objects as CSV",
        data=csv,
        file_name="visible_sky_objects.csv",
        mime="text/csv"
    )


# ==========================================================
# PROJECT NOTES
# ==========================================================

with st.expander("Scientific notes and limitations"):
    st.markdown(
        """
        This project uses:

        - **Skyfield + DE440** for Solar System positions.
        - **Hipparcos catalog** for stars.
        - **Alt/Az projection** from observer latitude, longitude, and time.
        - **Magnitude-based marker sizing** for stars and DSOs.
        - **Moon illumination approximation** using Sun-Moon angular separation.
        - **Twilight classification** using Sun altitude.
        - **Observation score** based on altitude, magnitude, Moon separation, darkness, and Bortle class.

        Current limitations:

        - Atmospheric refraction is not deeply modeled.
        - Light pollution is user-estimated via Bortle class.
        - DSO catalog is a curated Messier subset; expand it with OpenNGC or full Messier data later.
        - Constellation lines use a built-in fallback subset, not the full IAU sky boundary system.
        """
    )