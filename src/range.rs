//! Shared range-geometry validation and indexing.

use crate::error::{RadrsError, Result};
use nexrad_model::data::DataMoment;

/// Physical geometry of one NEXRAD data moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RangeGeometry {
    pub gate_count: usize,
    pub first_gate_m: u32,
    pub gate_spacing_m: u32,
}

/// Convert a decoded NEXRAD range value to its integral metre representation.
pub(crate) fn meters(value_km: f64, name: &str) -> Result<u32> {
    if !value_km.is_finite() || value_km < 0.0 {
        return Err(RadrsError::InvalidInput(format!(
            "{} must be a finite non-negative range, got {} km",
            name, value_km
        )));
    }
    let value_m = value_km * 1000.0;
    let rounded = value_m.round();
    if (value_m - rounded).abs() > 1e-6 || rounded > u32::MAX as f64 {
        return Err(RadrsError::InvalidInput(format!(
            "{} must be representable in integral metres, got {} km",
            name, value_km
        )));
    }
    Ok(rounded as u32)
}

pub(crate) fn geometry<M: DataMoment>(moment: &M) -> Result<RangeGeometry> {
    let gate_spacing_m = meters(moment.gate_interval_km(), "gate spacing")?;
    if gate_spacing_m == 0 {
        return Err(RadrsError::InvalidInput(
            "gate spacing must be greater than zero".into(),
        ));
    }
    let gate_count = moment.gate_count() as usize;
    if gate_count == 0 {
        return Err(RadrsError::InvalidInput(
            "moment gate count must be greater than zero".into(),
        ));
    }
    Ok(RangeGeometry {
        gate_count,
        first_gate_m: meters(moment.first_gate_range_km(), "first gate")?,
        gate_spacing_m,
    })
}

/// Return the physical center of a gate.
pub(crate) fn gate_center_m(geometry: RangeGeometry, gate: usize) -> Result<u32> {
    let offset = geometry
        .gate_spacing_m
        .checked_mul(gate as u32)
        .ok_or_else(|| RadrsError::InvalidInput("gate range exceeds metre range".into()))?;
    geometry
        .first_gate_m
        .checked_add(offset)
        .ok_or_else(|| RadrsError::InvalidInput("gate range exceeds metre range".into()))
}

/// Build the smallest regular metre lattice containing all moment grids.
///
/// A shared raystack range axis can only represent moments whose centers lie on
/// one regular lattice.  The grid starts at the nearest source gate and uses
/// the greatest common divisor of all source spacings.  Origins that are not
/// congruent on that lattice are rejected instead of shifting data by index.
pub(crate) fn canonical_lattice(geometries: &[RangeGeometry]) -> Result<Option<RangeGeometry>> {
    let Some(first) = geometries.first().copied() else {
        return Ok(None);
    };
    let mut spacing = first.gate_spacing_m;
    let mut first_gate_m = first.first_gate_m;
    let mut last_gate_m = gate_center_m(first, first.gate_count.saturating_sub(1))?;

    for geometry in geometries.iter().copied().skip(1) {
        spacing = gcd(spacing, geometry.gate_spacing_m);
        first_gate_m = first_gate_m.min(geometry.first_gate_m);
        last_gate_m = last_gate_m.max(gate_center_m(
            geometry,
            geometry.gate_count.saturating_sub(1),
        )?);
    }

    // Different first-gate origins are part of the lattice definition too.
    // Include their offsets in the gcd so grids such as 2,500 m + 1,000 m
    // and 2,125 m + 250 m resolve to an exact 125 m output lattice.
    for geometry in geometries.iter().copied() {
        spacing = gcd(spacing, geometry.first_gate_m - first_gate_m);
    }

    for geometry in geometries.iter().copied() {
        if (geometry.first_gate_m - first_gate_m) % spacing != 0 {
            return Err(RadrsError::InvalidInput(format!(
                "moment range geometry is not representable on one regular grid: first gates {} m and {} m, spacing {} m",
                first_gate_m, geometry.first_gate_m, spacing
            )));
        }
        let end = gate_center_m(geometry, geometry.gate_count.saturating_sub(1))?;
        if (end - first_gate_m) % spacing != 0 {
            return Err(RadrsError::InvalidInput(format!(
                "moment range geometry is not representable on one regular grid: gate at {} m is not aligned to origin {} m / spacing {} m",
                end, first_gate_m, spacing
            )));
        }
    }

    let gate_count = ((last_gate_m - first_gate_m) / spacing) as usize + 1;
    Ok(Some(RangeGeometry {
        gate_count,
        first_gate_m,
        gate_spacing_m: spacing,
    }))
}

pub(crate) fn map_gate_to_lattice(
    source: RangeGeometry,
    gate: usize,
    target: RangeGeometry,
) -> Result<usize> {
    let center = gate_center_m(source, gate)?;
    if center < target.first_gate_m
        || !(center - target.first_gate_m).is_multiple_of(target.gate_spacing_m)
    {
        return Err(RadrsError::InvalidInput(format!(
            "source gate at {} m cannot be represented on output range grid (origin {} m, spacing {} m)",
            center, target.first_gate_m, target.gate_spacing_m
        )));
    }
    let index = ((center - target.first_gate_m) / target.gate_spacing_m) as usize;
    if index >= target.gate_count {
        return Err(RadrsError::InvalidInput(format!(
            "source gate index {} falls outside output range grid of {} gates",
            index, target.gate_count
        )));
    }
    Ok(index)
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(gate_count: usize, first_gate_m: u32, gate_spacing_m: u32) -> RangeGeometry {
        RangeGeometry {
            gate_count,
            first_gate_m,
            gate_spacing_m,
        }
    }

    #[test]
    fn canonical_lattice_maps_aligned_mixed_spacing() {
        let target = canonical_lattice(&[grid(3, 2_000, 1_000), grid(9, 2_000, 250)])
            .unwrap()
            .unwrap();
        assert_eq!(target, grid(9, 2_000, 250));
        assert_eq!(
            map_gate_to_lattice(grid(3, 2_000, 1_000), 2, target).unwrap(),
            8
        );
    }

    #[test]
    fn canonical_lattice_includes_origin_offsets() {
        let surveillance = grid(334, 2_500, 1_000);
        let doppler = grid(912, 2_125, 250);
        let target = canonical_lattice(&[surveillance, doppler])
            .unwrap()
            .unwrap();
        assert_eq!(target, grid(2_668, 2_125, 125));
        assert_eq!(map_gate_to_lattice(surveillance, 0, target).unwrap(), 3);
        assert_eq!(
            map_gate_to_lattice(surveillance, 333, target).unwrap(),
            2_667
        );
        assert_eq!(map_gate_to_lattice(doppler, 0, target).unwrap(), 0);
        assert_eq!(map_gate_to_lattice(doppler, 911, target).unwrap(), 1_822);
    }
}
