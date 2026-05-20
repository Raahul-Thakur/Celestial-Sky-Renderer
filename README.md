# Celestial Sky Renderer

Interactive Streamlit app for rendering an observer-centered night sky map with stars, planets, the Sun, the Moon, selected Messier deep-sky objects, constellation guide lines, and observation scores.

## Features

- Observer location by city lookup or manual latitude/longitude.
- Timezone, date, time, and hour-offset controls.
- Skyfield-based solar system positions using the `de440.bsp` ephemeris.
- Hipparcos star catalog rendering with magnitude filters.
- Selected Messier deep-sky objects with visibility scoring.
- Moon illumination, Moon separation, Sun altitude, twilight, and darkness metrics.
- Plotly dark-sky Alt/Az visualization.
- Export table for visible objects.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- A local Skyfield ephemeris file named `de440.bsp` in the project root

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Add the ephemeris file:

```text
de440.bsp
```

Place it in the same directory as `app.py`. Skyfield ephemeris files are available from NASA/JPL through Skyfield's data download flow.

## Run

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal.

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```

## Notes

- The deep-sky object catalog is a curated Messier subset and can be expanded later.
- Constellation lines are built-in guide lines for a small set of recognizable patterns, not full IAU constellation boundaries.
- City search uses `geopy` with Nominatim, so geocoding requires network access.
