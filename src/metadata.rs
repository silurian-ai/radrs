//! Shared metadata extraction for NEXRAD volumes.

use nexrad_data::volume::File as VolumeFile;
use nexrad_decode::messages::MessageContents;

/// Basic scan metadata shared across xradar/raystack outputs.
#[derive(Default, Clone)]
pub struct ScanMeta {
    pub instrument_name: Option<String>,
    pub latitude: Option<f32>,
    pub longitude: Option<f32>,
    pub altitude: Option<f32>,
}

/// Extract scan-level metadata from a NEXRAD volume.
pub fn extract_scan_meta(volume: &VolumeFile) -> ScanMeta {
    let mut meta = ScanMeta::default();

    meta.instrument_name = volume
        .header()
        .and_then(|h| h.icao_of_radar())
        .map(|name| name.trim_matches('\0').trim().to_string());

    let records = match volume.records() {
        Ok(records) => records,
        Err(err) => {
            tracing::warn!("Failed to read volume records for metadata: {}", err);
            return meta;
        }
    };

    for record in records {
        let record = if record.compressed() {
            match record.decompress() {
                Ok(decompressed) => decompressed,
                Err(err) => {
                    tracing::warn!("Failed to decompress record for metadata: {}", err);
                    continue;
                }
            }
        } else {
            record
        };

        let messages = match record.messages() {
            Ok(messages) => messages,
            Err(err) => {
                tracing::warn!("Failed to decode record messages for metadata: {}", err);
                continue;
            }
        };

        for message in messages {
            if let MessageContents::DigitalRadarData(radar_data) = message.into_contents() {
                if let Some(volume_block) = radar_data.volume_data_block() {
                    let site_height = volume_block.site_height_raw() as f32;
                    let feedhorn_height = volume_block.feedhorn_height_raw() as f32;
                    meta.latitude = Some(volume_block.latitude_raw());
                    meta.longitude = Some(volume_block.longitude_raw());
                    meta.altitude = Some(site_height + feedhorn_height);
                    return meta;
                }
            }
        }
    }

    meta
}
