//! Scan parsing utilities

#![allow(dead_code)]

use crate::error::Result;
use nexrad_data::volume::File as VolumeFile;
use nexrad_model::data::Scan;

/// Parse a volume file into a Scan
pub fn parse_volume(data: &[u8]) -> Result<Scan> {
    let volume = VolumeFile::new(data.to_vec());
    let scan = volume.scan()?;
    Ok(scan)
}

/// Get volume header information
pub fn parse_header(data: &[u8]) -> Option<VolumeHeader> {
    let volume = VolumeFile::new(data.to_vec());
    let header = volume.header()?;

    // Version is extracted from tape_filename: "AR2V0 0xx." where xx is version
    let version = header
        .tape_filename()
        .and_then(|name| {
            // Format is "AR2V0 0xx." - version is at chars 7-8
            if name.len() >= 9 {
                name[7..9].parse::<u8>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);

    Some(VolumeHeader {
        version,
        icao: header.icao_of_radar(),
        datetime: header.date_time(),
    })
}

/// Volume header information
#[derive(Debug, Clone)]
pub struct VolumeHeader {
    pub version: u8,
    pub icao: Option<String>,
    pub datetime: Option<chrono::DateTime<chrono::Utc>>,
}
