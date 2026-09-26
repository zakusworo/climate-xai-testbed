"""FAO-56 Penman-Monteith reference evapotranspiration (daily step).

Implements Allen, Pereira, Raes & Smith (1998), FAO Irrigation and Drainage
Paper 56, chapter 3. Equation numbers refer to that paper. All inputs are
NumPy-broadcastable so the function can be used as a vectorised "true model"
inside SHAP explainers.

Units: temperatures in degC, relative humidity in %, wind speed at 2 m in m/s,
shortwave radiation Rs in MJ m-2 day-1, pressure in kPa, latitude in degrees.
Output: ET0 in mm day-1.
"""

from __future__ import annotations

import numpy as np

SIGMA = 4.903e-9  # Stefan-Boltzmann, MJ K-4 m-2 day-1
GSC = 0.0820  # solar constant, MJ m-2 min-1
ALBEDO = 0.23  # hypothetical grass reference crop


def sat_vapour_pressure(t):
    """Eq. 11, kPa."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def extraterrestrial_radiation(lat_deg, doy):
    """Eq. 21-25, Ra in MJ m-2 day-1."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    return (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def et0(tmax, tmin, rh, u2, rs, ps, lat, doy, elev=None, ea=None):
    """Daily FAO-56 PM ET0 (eq. 6).

    ``ps`` is surface pressure in kPa. If it is None, it is derived from
    ``elev`` (eq. 7). ``ea`` overrides the RHmean-based actual vapour
    pressure (eq. 19) when a measured value is available.
    """
    tmax, tmin, rh, u2, rs = map(np.asarray, (tmax, tmin, rh, u2, rs))
    tmean = (tmax + tmin) / 2
    if ps is None:
        ps = 101.3 * ((293 - 0.0065 * np.asarray(elev)) / 293) ** 5.26
    gamma = 0.665e-3 * np.asarray(ps)  # eq. 8
    delta = 4098 * sat_vapour_pressure(tmean) / (tmean + 237.3) ** 2  # eq. 13
    es = (sat_vapour_pressure(tmax) + sat_vapour_pressure(tmin)) / 2  # eq. 12
    if ea is None:
        ea = rh / 100 * es  # eq. 19
    ra = extraterrestrial_radiation(lat, doy)
    z = 0.0 if elev is None else np.asarray(elev)
    rso = (0.75 + 2e-5 * z) * ra  # eq. 37
    rns = (1 - ALBEDO) * rs  # eq. 38
    ratio = np.clip(rs / np.where(rso > 0, rso, np.nan), 0.25, 1.0)
    rnl = (
        SIGMA
        * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4)
        / 2
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0)))
        * (1.35 * ratio - 0.35)
    )  # eq. 39
    rn = rns - rnl
    g = 0.0  # daily soil heat flux, eq. 42
    num = 0.408 * delta * (rn - g) + gamma * 900 / (tmean + 273) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return num / den
