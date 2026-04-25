//! Cloud storage access for NEXRAD data
//!
//! Supports S3, GCS, Azure Blob Storage, and local filesystems.

use crate::error::RadrsError;
use crate::error::Result;
use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::parse_url_opts;
use object_store::aws::AmazonS3Builder;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use url::Url;

pub mod archive;
pub mod realtime;

pub use archive::fetch_s3_url;
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
    // Configure HTTP client for high throughput:
    // - Larger connection pool per host for parallel downloads
    // - HTTP/1.1 (faster than HTTP/2 for object storage per object_store benchmarks)
    let client_options = ClientOptions::new()
        .with_pool_max_idle_per_host(100)
        .with_pool_idle_timeout(std::time::Duration::from_secs(60));

    let store = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region("us-east-1")
        .with_skip_signature(true)
        .with_client_options(client_options)
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
    let url = parse_store_url(uri)?;
    let options = normalize_storage_options(options);
    let (store, _path) = parse_url_opts(&url, options)?;
    Ok(Arc::from(store))
}

fn parse_store_url(uri: &str) -> Result<Url> {
    if uri.starts_with("file://") {
        return Url::parse(uri)
            .map_err(|e| RadrsError::InvalidUrl(format!("Invalid file URI: {} ({})", uri, e)));
    }

    // No `://` → local filesystem path (absolute or relative). Url::from_file_path
    // requires an absolute path, so resolve relatives against the cwd first.
    if !uri.contains("://") {
        let path = std::path::Path::new(uri);
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| {
                    RadrsError::InvalidUrl(format!(
                        "Cannot resolve relative path {}: {}",
                        uri, e
                    ))
                })?
                .join(path)
        };
        return Url::from_file_path(&abs)
            .map_err(|_| RadrsError::InvalidUrl(format!("Invalid local path: {}", uri)));
    }

    let (scheme, rest) = uri
        .split_once("://")
        .ok_or_else(|| RadrsError::InvalidUrl(format!("Unsupported URI scheme: {}", uri)))?;
    let scheme = scheme.to_ascii_lowercase();

    match scheme.as_str() {
        "s3" | "gs" | "az" | "azure" => {
            let normalized = format!("{}://{}", scheme, rest);
            Url::parse(&normalized)
                .map_err(|e| RadrsError::InvalidUrl(format!("Invalid URI: {} ({})", uri, e)))
        }
        _ => Err(RadrsError::InvalidUrl(format!(
            "Unsupported URI scheme: {}",
            uri
        ))),
    }
}

fn normalize_storage_options(
    options: Option<HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut options = options.unwrap_or_default();

    // Backward-compatible alias for existing Python/Rust callers:
    // storage_options={"anon":"true"} maps to object_store's skip_signature.
    if let Some(anon) = options.remove("anon") {
        let skip_key_present = options.contains_key("skip_signature")
            || options.contains_key("aws_skip_signature")
            || options.contains_key("google_skip_signature")
            || options.contains_key("azure_skip_signature");

        if anon == "true" && !skip_key_present {
            options.insert("skip_signature".to_string(), "true".to_string());
        }
    }

    options
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

/// Fetch bytes from a URL using object_store
///
/// Supports S3, GCS, Azure Blob Storage, and local filesystems.
///
/// # Arguments
/// * `url` - Full URL to the object (e.g., "s3://bucket/path/to/file", "/local/path/file")
/// * `storage_options` - Optional storage credentials and configuration
///
/// # Examples
///
/// ```ignore
/// // S3 with anonymous access
/// let bytes = fetch_bytes_from_url(
///     "s3://noaa-nexrad-level2/2024/03/15/KTLX/KTLX20240315_120000_V06",
///     Some(hashmap!{"anon" => "true"})
/// ).await?;
///
/// // GCS with service account
/// let bytes = fetch_bytes_from_url(
///     "gs://my-bucket/nexrad/KTLX20240315_120000_V06",
///     Some(hashmap!{"service_account_path" => "/path/to/key.json"})
/// ).await?;
///
/// // Local filesystem
/// let bytes = fetch_bytes_from_url("/data/nexrad/KTLX20240315_120000_V06", None).await?;
/// ```
pub async fn fetch_bytes_from_url(
    url: &str,
    storage_options: Option<HashMap<String, String>>,
) -> Result<bytes::Bytes> {
    let parsed_url = parse_store_url(url)?;
    let options = normalize_storage_options(storage_options);
    let (store, path) = parse_url_opts(&parsed_url, options)?;
    let store: Arc<dyn ObjectStore> = Arc::from(store);
    if path.as_ref().is_empty() {
        return Err(RadrsError::InvalidUrl(format!(
            "No object path in URL: {}",
            url
        )));
    }

    // Fetch the object
    let get_result = store.get(&path).await?;
    let bytes = get_result.bytes().await?;

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_store_url_cloud_and_local() {
        assert_eq!(
            parse_store_url("S3://my-bucket/path/file")
                .unwrap()
                .scheme(),
            "s3"
        );
        assert_eq!(
            parse_store_url("gs://my-bucket/path/file")
                .unwrap()
                .scheme(),
            "gs"
        );
        assert_eq!(
            parse_store_url("azure://my-container/path/file")
                .unwrap()
                .scheme(),
            "azure"
        );
        assert_eq!(
            parse_store_url("/tmp/test-file")
                .unwrap()
                .scheme(),
            "file"
        );
        assert!(parse_store_url("http://example.com").is_err());
    }

    #[test]
    fn test_parse_store_url_relative_path() {
        // Relative path resolves against cwd to a file:// URL.
        let url = parse_store_url("relative/path/file.ar2v").unwrap();
        assert_eq!(url.scheme(), "file");
        let cwd = std::env::current_dir().unwrap();
        assert!(url.path().starts_with(cwd.to_str().unwrap()));
        assert!(url.path().ends_with("relative/path/file.ar2v"));

        // Bare filename works too.
        let url = parse_store_url("file.ar2v").unwrap();
        assert_eq!(url.scheme(), "file");
        assert!(url.path().ends_with("file.ar2v"));
    }

    #[test]
    fn test_storage_options_anon_alias() {
        let mut input = HashMap::new();
        input.insert("anon".to_string(), "true".to_string());
        let normalized = normalize_storage_options(Some(input));
        assert_eq!(
            normalized.get("skip_signature").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn test_storage_options_explicit_skip_signature_wins() {
        let mut input = HashMap::new();
        input.insert("anon".to_string(), "true".to_string());
        input.insert("skip_signature".to_string(), "false".to_string());
        let normalized = normalize_storage_options(Some(input));
        assert_eq!(
            normalized.get("skip_signature").map(String::as_str),
            Some("false")
        );
    }

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
