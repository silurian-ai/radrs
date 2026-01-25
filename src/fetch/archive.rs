//! Archive data access from S3

use crate::error::{RadrsError, Result};
use crate::fetch::{ARCHIVE_STORE, FETCH_SEMAPHORE, store_for_bucket};
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use std::sync::Arc;

/// Fetch a file from the NEXRAD archive
pub async fn fetch_archive_file(
    site: &str,
    year: i32,
    month: u32,
    day: u32,
    filename: &str,
) -> Result<Vec<u8>> {
    let path = format!("{}/{:02}/{:02}/{}/{}", year, month, day, site, filename);
    let object_path = ObjectPath::from(path);

    fetch_object_bytes(&ARCHIVE_STORE, &object_path).await
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
    fetch_object_bytes(&store, &path).await
}

async fn fetch_object_bytes(store: &Arc<dyn ObjectStore>, path: &ObjectPath) -> Result<Vec<u8>> {
    let _permit = FETCH_SEMAPHORE.acquire().await.expect("semaphore closed");
    let result = store.get(path).await?;
    let bytes = result.bytes().await?;
    Ok(bytes.to_vec())
}
