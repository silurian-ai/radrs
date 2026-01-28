//! Cloud storage access for NEXRAD data
//!
//! Supports S3, GCS, Azure Blob Storage, and local filesystems.

use crate::error::{RadrsError, Result};
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

pub mod archive;
pub mod realtime;

pub use archive::{fetch_archive_file, fetch_s3_url};
pub use realtime::poll_realtime_chunks;

/// Public archive bucket for complete NEXRAD Level 2 volumes.
pub const ARCHIVE_BUCKET: &str = "unidata-nexrad-level2";

/// Global Tokio runtime (reuse across calls).
pub static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("failed to create global tokio runtime"));

/// Limit concurrent S3 requests.
pub static FETCH_SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(50)));

/// Shared archive S3 client.
pub static ARCHIVE_STORE: Lazy<Arc<dyn ObjectStore>> =
    Lazy::new(|| build_store(ARCHIVE_BUCKET).expect("failed to create archive S3 client"));

static STORE_CACHE: Lazy<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(ARCHIVE_BUCKET.to_string(), ARCHIVE_STORE.clone());
    Mutex::new(map)
});

fn build_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    let store = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region("us-east-1")
        .with_skip_signature(true)
        .build()?;

    Ok(Arc::new(store))
}

/// Get (or create) a shared S3 client for a bucket.
pub fn store_for_bucket(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    if let Some(store) = STORE_CACHE
        .lock()
        .expect("store cache poisoned")
        .get(bucket)
        .cloned()
    {
        return Ok(store);
    }

    let store = build_store(bucket)?;
    STORE_CACHE
        .lock()
        .expect("store cache poisoned")
        .insert(bucket.to_string(), store.clone());
    Ok(store)
}

/// Build an ObjectStore from a URI with optional storage options
///
/// Supports S3, GCS, Azure, and local filesystem URIs.
///
/// # Examples
///
/// ```ignore
/// // S3 with anonymous access
/// let store = build_store_from_uri("s3://noaa-nexrad-level2", Some(hashmap!{
///     "anon".to_string() => "true".to_string()
/// }))?;
///
/// // GCS with service account
/// let store = build_store_from_uri("gs://my-bucket", Some(hashmap!{
///     "service_account_path".to_string() => "/path/to/key.json".to_string()
/// }))?;
///
/// // Local filesystem
/// let store = build_store_from_uri("/data/nexrad", None)?;
/// ```
pub fn build_store_from_uri(
    uri: &str,
    options: Option<HashMap<String, String>>,
) -> Result<Arc<dyn ObjectStore>> {
    let options = options.unwrap_or_default();

    if uri.starts_with("s3://") || uri.starts_with("S3://") {
        build_s3_store(uri, options)
    } else if uri.starts_with("gs://") || uri.starts_with("GS://") {
        build_gcs_store(uri, options)
    } else if uri.starts_with("az://") || uri.starts_with("AZ://") || uri.starts_with("azure://") {
        build_azure_store(uri, options)
    } else if uri.starts_with("file://") || uri.starts_with('/') {
        build_local_store(uri, options)
    } else {
        Err(RadrsError::InvalidUrl(format!(
            "Unsupported URI scheme: {}",
            uri
        )))
    }
}

fn build_s3_store(uri: &str, options: HashMap<String, String>) -> Result<Arc<dyn ObjectStore>> {
    let url_without_scheme = uri
        .strip_prefix("s3://")
        .or_else(|| uri.strip_prefix("S3://"))
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Invalid S3 URI: {}", uri)))?;

    let bucket = url_without_scheme
        .split('/')
        .next()
        .ok_or_else(|| RadrsError::InvalidUrl(format!("No bucket in S3 URI: {}", uri)))?;

    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket);

    // Apply storage options
    if let Some(region) = options.get("region") {
        builder = builder.with_region(region);
    }
    if let Some(endpoint) = options.get("endpoint") {
        builder = builder.with_endpoint(endpoint);
    }
    if let Some(access_key_id) = options.get("access_key_id") {
        builder = builder.with_access_key_id(access_key_id);
    }
    if let Some(secret_access_key) = options.get("secret_access_key") {
        builder = builder.with_secret_access_key(secret_access_key);
    }
    if let Some(token) = options.get("token") {
        builder = builder.with_token(token);
    }
    if options.get("anon").map(|s| s == "true").unwrap_or(false) {
        builder = builder.with_skip_signature(true);
    }

    let store = builder.build()?;
    Ok(Arc::new(store))
}

fn build_gcs_store(uri: &str, options: HashMap<String, String>) -> Result<Arc<dyn ObjectStore>> {
    use object_store::gcp::GoogleCloudStorageBuilder;

    let url_without_scheme = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("GS://"))
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Invalid GCS URI: {}", uri)))?;

    let bucket = url_without_scheme
        .split('/')
        .next()
        .ok_or_else(|| RadrsError::InvalidUrl(format!("No bucket in GCS URI: {}", uri)))?;

    let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(bucket);

    // Apply storage options
    if let Some(service_account_path) = options.get("service_account_path") {
        builder = builder.with_service_account_path(service_account_path);
    }
    if let Some(service_account_key) = options.get("service_account_key") {
        builder = builder.with_service_account_key(service_account_key);
    }

    let store = builder.build()?;
    Ok(Arc::new(store))
}

fn build_azure_store(uri: &str, options: HashMap<String, String>) -> Result<Arc<dyn ObjectStore>> {
    use object_store::azure::MicrosoftAzureBuilder;

    let url_without_scheme = uri
        .strip_prefix("az://")
        .or_else(|| uri.strip_prefix("AZ://"))
        .or_else(|| uri.strip_prefix("azure://"))
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Invalid Azure URI: {}", uri)))?;

    let container = url_without_scheme
        .split('/')
        .next()
        .ok_or_else(|| RadrsError::InvalidUrl(format!("No container in Azure URI: {}", uri)))?;

    let mut builder = MicrosoftAzureBuilder::new().with_container_name(container);

    // Apply storage options
    if let Some(account_name) = options.get("account_name") {
        builder = builder.with_account(account_name);
    }
    if let Some(access_key) = options.get("access_key") {
        builder = builder.with_access_key(access_key);
    }
    if let Some(bearer_token) = options.get("bearer_token") {
        builder = builder.with_bearer_token_authorization(bearer_token);
    }

    let store = builder.build()?;
    Ok(Arc::new(store))
}

fn build_local_store(_uri: &str, _options: HashMap<String, String>) -> Result<Arc<dyn ObjectStore>> {
    use object_store::local::LocalFileSystem;

    // LocalFileSystem uses the entire filesystem as root
    let store = LocalFileSystem::new();
    Ok(Arc::new(store))
}

/// Extract object path from a full URL
///
/// Strips the scheme and bucket/container, returning just the object path.
///
/// # Examples
/// - s3://bucket/path/to/file.txt -> path/to/file.txt
/// - gs://bucket/data/file.txt -> data/file.txt
/// - az://container/path/file.txt -> path/file.txt
/// - /local/path/file.txt -> /local/path/file.txt
/// - file:///path/to/file.txt -> /path/to/file.txt
pub(crate) fn extract_object_path_from_url(url: &str) -> Result<String> {
    if let Some(rest) = url
        .strip_prefix("s3://")
        .or_else(|| url.strip_prefix("S3://"))
        .or_else(|| url.strip_prefix("gs://"))
        .or_else(|| url.strip_prefix("GS://"))
        .or_else(|| url.strip_prefix("az://"))
        .or_else(|| url.strip_prefix("AZ://"))
        .or_else(|| url.strip_prefix("azure://"))
    {
        // Split on first '/' to remove bucket/container
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        if parts.len() == 2 {
            return Ok(parts[1].to_string());
        } else {
            return Err(RadrsError::InvalidUrl(format!(
                "No object path in URL: {}",
                url
            )));
        }
    }

    if let Some(rest) = url.strip_prefix("file://") {
        return Ok(rest.to_string());
    }

    if url.starts_with('/') {
        return Ok(url.to_string());
    }

    Err(RadrsError::InvalidUrl(format!(
        "Unsupported URL scheme: {}",
        url
    )))
}

/// Extract base path from a URI (remove scheme and bucket/container)
///
/// Examples:
/// - s3://noaa-nexrad-level2/2024/03/15 -> "2024/03/15"
/// - gs://my-bucket/nexrad/data -> "nexrad/data"
/// - az://container/path -> "path"
/// - /local/path/data -> "/local/path/data"
/// - file:///local/path -> "/local/path"
pub(crate) fn extract_base_path(uri: &str) -> Result<String> {
    if let Some(rest) = uri
        .strip_prefix("s3://")
        .or_else(|| uri.strip_prefix("S3://"))
    {
        // s3://bucket/path -> path (or empty if just bucket)
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        return Ok(if parts.len() == 2 {
            parts[1].to_string()
        } else {
            String::new()
        });
    }

    if let Some(rest) = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("GS://"))
    {
        // gs://bucket/path -> path
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        return Ok(if parts.len() == 2 {
            parts[1].to_string()
        } else {
            String::new()
        });
    }

    if let Some(rest) = uri
        .strip_prefix("az://")
        .or_else(|| uri.strip_prefix("AZ://"))
        .or_else(|| uri.strip_prefix("azure://"))
    {
        // az://container/path -> path
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        return Ok(if parts.len() == 2 {
            parts[1].to_string()
        } else {
            String::new()
        });
    }

    if let Some(rest) = uri.strip_prefix("file://") {
        // file:///path -> /path
        return Ok(rest.to_string());
    }

    if uri.starts_with('/') {
        // /local/path -> /local/path
        return Ok(uri.to_string());
    }

    Err(RadrsError::InvalidUrl(format!(
        "Unsupported URI scheme: {}",
        uri
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_base_path() {
        // S3
        assert_eq!(
            extract_base_path("s3://noaa-nexrad-level2/2024/03/15").unwrap(),
            "2024/03/15"
        );
        assert_eq!(extract_base_path("s3://my-bucket").unwrap(), "");

        // GCS
        assert_eq!(
            extract_base_path("gs://my-bucket/nexrad/data").unwrap(),
            "nexrad/data"
        );
        assert_eq!(extract_base_path("gs://bucket").unwrap(), "");

        // Azure
        assert_eq!(
            extract_base_path("az://container/path/to/data").unwrap(),
            "path/to/data"
        );
        assert_eq!(extract_base_path("azure://container").unwrap(), "");

        // Local filesystem
        assert_eq!(
            extract_base_path("/local/path/data").unwrap(),
            "/local/path/data"
        );
        assert_eq!(
            extract_base_path("file:///data/nexrad").unwrap(),
            "/data/nexrad"
        );

        // Unsupported
        assert!(extract_base_path("http://example.com").is_err());
        assert!(extract_base_path("invalid").is_err());
    }
}
