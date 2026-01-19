//! Archive data access from S3

use crate::error::{RadrsError, Result};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;

const ARCHIVE_BUCKET: &str = "unidata-nexrad-level2";

/// Fetch a file from the NEXRAD archive
pub async fn fetch_archive_file(
    site: &str,
    year: i32,
    month: u32,
    day: u32,
    filename: &str,
) -> Result<Vec<u8>> {
    let store = AmazonS3Builder::new()
        .with_bucket_name(ARCHIVE_BUCKET)
        .with_region("us-east-1")
        .with_skip_signature(true)
        .build()?;

    let path = format!("{}/{:02}/{:02}/{}/{}", year, month, day, site, filename);
    let object_path = ObjectPath::from(path);

    let result = store.get(&object_path).await?;
    let bytes = result.bytes().await?;

    Ok(bytes.to_vec())
}

/// Parse an S3 URL and fetch the file
pub async fn fetch_s3_url(url: &str) -> Result<Vec<u8>> {
    // Parse s3://bucket/path format
    let url = url.strip_prefix("s3://").ok_or_else(|| {
        RadrsError::InvalidUrl(format!("Not an S3 URL: {}", url))
    })?;

    let (bucket, key) = url.split_once('/').ok_or_else(|| {
        RadrsError::InvalidUrl(format!("Invalid S3 URL format: s3://{}", url))
    })?;

    let store = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region("us-east-1")
        .with_skip_signature(true)
        .build()?;

    let path = ObjectPath::from(key);
    let result = store.get(&path).await?;
    let bytes = result.bytes().await?;

    Ok(bytes.to_vec())
}
