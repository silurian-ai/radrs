//! S3 access for NEXRAD data (internal)

use crate::error::Result;
use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;
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
pub static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("failed to create global tokio runtime")
});

/// Limit concurrent S3 requests.
pub static FETCH_SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(50)));

/// Shared archive S3 client.
pub static ARCHIVE_STORE: Lazy<Arc<dyn ObjectStore>> = Lazy::new(|| {
    build_store(ARCHIVE_BUCKET).expect("failed to create archive S3 client")
});

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
