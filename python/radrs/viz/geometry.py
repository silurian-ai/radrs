"""Beam geometry: slant range and elevation to ground range and height."""

from __future__ import annotations

from typing import Final

import numpy as np

#: Mean earth radius (meters).
EARTH_RADIUS_M: Final[float] = 6_371_000.0

#: Effective earth radius (meters) under the standard 4/3 refraction model.
EFFECTIVE_EARTH_RADIUS_M: Final[float] = 4.0 / 3.0 * EARTH_RADIUS_M


def beam_geometry(
    range_m: np.ndarray | float,
    elevation_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert slant range and elevation to ground range and height.

    Uses the standard 4/3 effective earth radius model (Doviak & Zrnic,
    *Doppler Radar and Weather Observations*), which folds mean atmospheric
    refraction into an inflated earth radius so the beam can be treated as a
    straight line above a more strongly curved earth:

    .. code-block:: text

        R_e = (4/3) * 6371 km
        h   = sqrt(r^2 + R_e^2 + 2 * r * R_e * sin(el)) - R_e
        s   = R_e * asin(r * cos(el) / (R_e + h))

    Assumptions
    -----------
    * *Standard refraction.* The 4/3 rule holds for a normally stratified
      atmosphere. Ducting, superrefraction, and strong inversions are not
      modeled and will bias heights low or high.
    * *Radar at height 0.* Returned heights are above the radar antenna, not
      above mean sea level. Add the antenna altitude to get MSL heights.

    Parameters
    ----------
    range_m : array_like
        Slant range along the beam, in meters. Scalars and arrays both work.
    elevation_deg : array_like
        Beam elevation angle in degrees, broadcast against ``range_m``.

    Returns
    -------
    ground_range_m : ndarray
        Great-circle distance along the earth's surface, in meters. Always
        slightly less than the slant range.
    height_m : ndarray
        Height above the radar antenna, in meters. Always slightly more than
        the flat-earth ``r * sin(el)``, and increasingly so with range.

    Examples
    --------
    >>> ground_range, height = beam_geometry(200_000.0, 0.5)
    >>> round(float(height))
    4099
    """

    r = np.asarray(range_m, dtype=np.float64)
    elevation_rad = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    r_e = EFFECTIVE_EARTH_RADIUS_M

    height_m = (
        np.sqrt(r * r + r_e * r_e + 2.0 * r * r_e * np.sin(elevation_rad)) - r_e
    )
    # Clip guards against float error pushing the ratio past 1 at extreme ranges.
    ratio = np.clip(r * np.cos(elevation_rad) / (r_e + height_m), -1.0, 1.0)
    ground_range_m = r_e * np.arcsin(ratio)
    return ground_range_m, height_m


def _polar_to_cartesian(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    range_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map polar gate coordinates to east/north/up meters via :func:`beam_geometry`."""

    azimuth_rad = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    horizontal, z = beam_geometry(range_m, elevation_deg)

    # x=east, y=north, z=up
    x = horizontal * np.sin(azimuth_rad)
    y = horizontal * np.cos(azimuth_rad)
    return x, y, z
