//! Shared constants for radar metadata and moment names.

use nexrad_model::data::Product;

/// CF conventions version used in exported datasets.
pub const CF_CONVENTIONS: &str = "CF-1.8";

/// Instrument type for NEXRAD sites.
pub const INSTRUMENT_TYPE: &str = "radar";

/// Platform type for NEXRAD sites.
pub const PLATFORM_TYPE: &str = "fixed";

/// Default sweep mode for NEXRAD volumes.
pub const DEFAULT_SWEEP_MODE: &str = "azimuth_surveillance";

/// Default PRT mode when not set in source data.
pub const DEFAULT_PRT_MODE: &str = "not_set";

/// Default follow mode when not set in source data.
pub const DEFAULT_FOLLOW_MODE: &str = "not_set";

/// Standard NEXRAD moment names (raystack/radar outputs).
pub const MOMENT_NAMES: [&str; 7] = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH"];

/// Mapping of NEXRAD moments to xradar/CF variable names, in `MOMENT_NAMES` order.
pub const XRADAR_MOMENT_NAMES: [(Product, &str); 7] = [
    (Product::Reflectivity, "DBZH"),
    (Product::Velocity, "VRADH"),
    (Product::SpectrumWidth, "WRADH"),
    (Product::DifferentialReflectivity, "ZDR"),
    (Product::DifferentialPhase, "PHIDP"),
    (Product::CorrelationCoefficient, "RHOHV"),
    (Product::ClutterFilterPower, "CCORH"),
];
