//! Shared constants for radar metadata and moment names.

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

/// Mapping of NEXRAD moments to xradar/CF variable names.
pub const XRADAR_MOMENT_NAMES: [(&str, &str); 7] = [
    ("reflectivity", "DBZH"),
    ("velocity", "VRADH"),
    ("spectrum_width", "WRADH"),
    ("differential_reflectivity", "ZDR"),
    ("differential_phase", "PHIDP"),
    ("correlation_coefficient", "RHOHV"),
    ("clutter_filter_power", "CCORH"),
];
