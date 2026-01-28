//! Moment data decoding utilities

#![allow(dead_code)]

use nexrad_model::data::{MomentData, MomentValue};

/// Moment type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MomentType {
    Reflectivity,
    Velocity,
    SpectrumWidth,
    DifferentialReflectivity,
    DifferentialPhase,
    CorrelationCoefficient,
    ClutterFilterPower,
}

impl MomentType {
    /// Get the CF-radial standard name for this moment type
    pub fn cf_name(&self) -> &'static str {
        match self {
            MomentType::Reflectivity => "DBZH",
            MomentType::Velocity => "VRADH",
            MomentType::SpectrumWidth => "WRADH",
            MomentType::DifferentialReflectivity => "ZDR",
            MomentType::DifferentialPhase => "PHIDP",
            MomentType::CorrelationCoefficient => "RHOHV",
            MomentType::ClutterFilterPower => "CCORH",
        }
    }

    /// Get all moment types
    pub fn all() -> &'static [MomentType] {
        &[
            MomentType::Reflectivity,
            MomentType::Velocity,
            MomentType::SpectrumWidth,
            MomentType::DifferentialReflectivity,
            MomentType::DifferentialPhase,
            MomentType::CorrelationCoefficient,
            MomentType::ClutterFilterPower,
        ]
    }
}

/// Decode moment data to f32 values
///
/// Returns NaN for below threshold and range folded values
pub fn decode_moment_values(moment: &MomentData) -> Vec<f32> {
    moment
        .values()
        .iter()
        .map(|v| match v {
            MomentValue::Value(x) => *x,
            MomentValue::BelowThreshold => f32::NAN,
            MomentValue::RangeFolded => f32::NAN,
            MomentValue::CfpStatus(_) => f32::NAN,
        })
        .collect()
}

/// Decode moment data with separate handling of special values
pub fn decode_moment_values_with_flags(moment: &MomentData) -> (Vec<f32>, Vec<u8>) {
    let mut values = Vec::with_capacity(moment.gate_count() as usize);
    let mut flags = Vec::with_capacity(moment.gate_count() as usize);

    for v in moment.values() {
        match v {
            MomentValue::Value(x) => {
                values.push(x);
                flags.push(0); // Valid
            }
            MomentValue::BelowThreshold => {
                values.push(f32::NAN);
                flags.push(1); // Below threshold
            }
            MomentValue::RangeFolded => {
                values.push(f32::NAN);
                flags.push(2); // Range folded
            }
            MomentValue::CfpStatus(_) => {
                values.push(f32::NAN);
                flags.push(3); // CFP status (non-numeric)
            }
        }
    }

    (values, flags)
}
