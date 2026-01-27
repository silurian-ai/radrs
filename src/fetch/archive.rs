//! Archive data access from S3

use crate::error::{RadrsError, Result};
use crate::fetch::{ARCHIVE_STORE, FETCH_SEMAPHORE, store_for_bucket};
use object_store::{ObjectStore, ObjectStoreExt};
use object_store::path::Path as ObjectPath;
use std::sync::Arc;
use std::time::Instant;

/// Fetch a file from the NEXRAD archive
pub async fn fetch_archive_file(
    site: &str,
    year: i32,
    month: u32,
    day: u32,
    filename: &str,
) -> Result<Vec<u8>> {
    let path = format!("{}/{:02}/{:02}/{}/{}", year, month, day, site, filename);
    let object_path = ObjectPath::from(path.as_str());
    let source = format!("archive://{}", path);

    fetch_object_bytes(&ARCHIVE_STORE, &object_path, &source).await
}

/// Parse an S3 URL and fetch the file
pub async fn fetch_s3_url(url: &str) -> Result<Vec<u8>> {
    // Parse s3://bucket/path format
    let url = url
        .strip_prefix("s3://")
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Not an S3 URL: {}", url)))?;

    let (bucket, key) = url
        .split_once('/')
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Invalid S3 URL format: s3://{}", url)))?;

    let store = store_for_bucket(bucket)?;
    let path = ObjectPath::from(key);
    let source = format!("s3://{}/{}", bucket, key);
    fetch_object_bytes(&store, &path, &source).await
}

async fn fetch_object_bytes(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectPath,
    source: &str,
) -> Result<Vec<u8>> {
    let start = Instant::now();
    let _permit = FETCH_SEMAPHORE.acquire().await.expect("semaphore closed");
    let result = match store.get(path).await {
        Ok(result) => result,
        Err(err) => {
            tracing::warn!(target: "radrs::fetch", source = source, error = %err);
            return Err(err.into());
        }
    };
    let bytes = match result.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(target: "radrs::fetch", source = source, error = %err);
            return Err(err.into());
        }
    };
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_millis() as u64;
    let bytes_len = bytes.len() as f64;
    let secs = elapsed.as_secs_f64();
    let mbps = if secs > 0.0 {
        (bytes_len * 8.0) / (secs * 1_000_000.0)
    } else {
        0.0
    };
    tracing::info!(
        target: "radrs::fetch",
        source = source,
        bytes = bytes.len(),
        elapsed_ms,
        mbps
    );
    Ok(bytes.to_vec())
}
