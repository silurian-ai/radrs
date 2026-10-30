//! LDM record decompression and volume decoding.
//!
//! This is the one place radrs turns an Archive II volume's compressed LDM
//! records into decoded messages. It mirrors `nexrad_data::volume::File::scan`
//! but swaps the `bzip2` crate (libbz2-rs-sys) for `recast_radar_bzip2`, a
//! dependency-free pure-Rust decoder tuned for NEXRAD records that is about
//! twice as fast single-threaded with identical output. The decoder owns
//! ~7 MiB of work buffers that are reused across calls, so one instance is
//! kept per thread (rayon workers included) in a thread-local.
//!
//! Every caller that needs decompressed records goes through
//! [`decompress_record`] or [`scan_volume`]; nothing else in the crate should
//! call `Record::decompress` or `File::scan` directly.
//!
//! Upstream nexrad-data declined to vendor this decoder
//! (<https://github.com/danielway/nexrad/pull/218>); an optional dependency
//! there becomes possible if `recast-radar-bzip2` is published to crates.io.

use crate::error::Result;
use nexrad_data::volume::{File as VolumeFile, Record};
use nexrad_decode::model::ScanBuilder;
use nexrad_model::data::Scan;
use rayon::prelude::*;
use recast_radar_bzip2::Decoder;
use std::cell::RefCell;

/// Upper bound on the decompressed size of one LDM record.
///
/// This is only a guard against corrupt or hostile input (a bzip2 block can
/// expand ~47 MB from a few bytes); a real Archive II record decompresses to a
/// few MiB. The `bzip2`-based path has no bound at all, so this is
/// deliberately generous rather than tight.
const MAX_RECORD_OUTPUT: usize = 256 << 20;

thread_local! {
    static DECODER: RefCell<Decoder> = RefCell::new({
        let mut decoder = Decoder::new();
        decoder.set_max_output(MAX_RECORD_OUTPUT);
        decoder
    });
}

/// Decompress one LDM record, matching `Record::decompress` semantics.
///
/// A record whose bytes 4..6 are `BZ` is a 4-byte big-endian size prefix
/// followed by one bzip2 stream; the prefix is skipped and the stream decoded.
/// Any other record is returned as an owned copy unchanged, so calling this on
/// a record of unknown state is safe.
pub fn decompress_record<'b>(record: &Record<'_>) -> Result<Record<'b>> {
    if !record.compressed() {
        return Ok(Record::new(record.data().to_vec()));
    }

    let input = &record.data()[4..];
    let mut out = Vec::new();
    DECODER.with(|decoder| decoder.borrow_mut().decode_stream_into(input, &mut out))?;
    Ok(Record::new(out))
}

/// Decode a volume file into a `Scan`, the same way `File::scan` does but with
/// the `recast_radar_bzip2` decoder.
///
/// A gzip-wrapped volume is inflated first (copying it once). Records are
/// decompressed and decoded in parallel on the rayon pool into one
/// `ScanBuilder` each, then merged; set `RAYON_NUM_THREADS=1` for a
/// single-threaded run.
pub fn scan_volume(volume: &VolumeFile) -> Result<Scan> {
    let inflated;
    let volume = if volume.compressed() {
        inflated = VolumeFile::new(volume.data().to_vec()).decompress()?;
        &inflated
    } else {
        volume
    };

    let builders = volume
        .records()?
        .into_par_iter()
        .map(|record| -> Result<ScanBuilder> {
            let record = decompress_record(&record)?;
            let mut builder = ScanBuilder::new();
            builder.add_messages(record.messages()?)?;
            Ok(builder)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut scan = ScanBuilder::new();
    for builder in builders {
        scan.merge(builder);
    }
    Ok(scan.build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RadrsError;

    #[test]
    fn uncompressed_record_is_copied_unchanged() {
        let bytes = [0, 0, 0, 4, b'X', b'Y', 1, 2];
        let record = Record::from_slice(&bytes);
        let out = decompress_record(&record).unwrap();
        assert_eq!(out.data(), &bytes);
    }

    #[test]
    fn corrupt_bzip2_stream_is_an_error() {
        // Valid "BZ" marker but garbage afterwards.
        let record =
            Record::from_slice(&[0, 0, 0, 8, b'B', b'Z', b'h', b'9', 0xff, 0xff, 0xff, 0xff]);
        let err = decompress_record(&record).unwrap_err();
        assert!(matches!(err, RadrsError::Bzip2(_)), "{err}");
    }
}
