"""NASA POWER daily data for Indonesian sites, cached as parquet.

NASA POWER is MERRA-2 / CERES reanalysis on a ~0.5 deg grid, not station
data. Coastal cells (Kupang, Makassar) are sea-influenced and show a small
diurnal temperature range. That is acceptable here because the testbed's
ground truth is the FAO-56 equation evaluated on these same inputs; it is not
a claim about measured evapotranspiration at the site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .fao56 import et0

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = {
    "T2M_MAX": "tmax",
    "T2M_MIN": "tmin",
    "T2M": "tmean_obs",
    "RH2M": "rh",
    "QV2M": "qv",
    "WS2M": "u2",
    "ALLSKY_SFC_SW_DWN": "rs",
    "PS": "ps",
    "PRECTOTCORR": "precip",
    "CLOUD_AMT": "cloud",
}


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    lat: float
    lon: float
    climate: str


SITES = [
    Site("sleman", "Sleman, DI Yogyakarta", -7.73, 110.36, "tropical monsoon, inland"),
    Site("malang", "Malang, East Java", -7.96, 112.62, "tropical highland"),
    Site("pontianak", "Pontianak, West Kalimantan", -0.03, 109.34, "equatorial rainforest"),
    Site("medan", "Medan, North Sumatra", 3.59, 98.67, "equatorial, northern hemisphere"),
    Site("makassar", "Makassar, South Sulawesi", -5.14, 119.43, "monsoon, coastal"),
    Site("kupang", "Kupang, East Nusa Tenggara", -10.17, 123.61, "semi-arid savanna, coastal"),
]
SITE_BY_KEY = {s.key: s for s in SITES}


def fetch_site(site: Site, start: int = 2011, end: int = 2024, force: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"power_{site.key}_{start}_{end}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    r = httpx.get(
        API,
        params={
            "parameters": ",".join(PARAMS),
            "community": "AG",
            "latitude": site.lat,
            "longitude": site.lon,
            "start": f"{start}0101",
            "end": f"{end}1231",
            "format": "JSON",
        },
        timeout=180,
    )
    r.raise_for_status()
    payload = r.json()
    p = payload["properties"]["parameter"]
    df = pd.DataFrame({PARAMS[k]: pd.Series(v) for k, v in p.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, np.nan)
    df["elev_power"] = payload["geometry"]["coordinates"][2]
    df.to_parquet(path)
    (CACHE / f"power_{site.key}_{start}_{end}.meta.json").write_text(
        json.dumps({"url": str(r.url), "header": payload.get("header", {})}, indent=2)
    )
    return df


def build_dataset(sites=SITES, start: int = 2011, end: int = 2024) -> pd.DataFrame:
    frames = []
    for s in sites:
        df = fetch_site(s, start, end).copy()
        df["site"] = s.key
        df["lat"] = s.lat
        df["doy"] = df.index.dayofyear
        frames.append(df)
    df = pd.concat(frames).dropna(subset=["tmax", "tmin", "rh", "u2", "rs", "ps"])
    df["et0"] = et0(df.tmax, df.tmin, df.rh, df.u2, df.rs, df.ps, df.lat, df.doy)
    return df
