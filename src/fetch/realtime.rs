//! Realtime data access from S3

use crate::error::Result;
use futures::stream::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use std::collections::HashSet;
use std::sync::Arc;

const REALTIME_BUCKET: &str = "unidata-nexrad-level2-chunks";

/// Chunk identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkId {
    pub site: String,
    pub volume_time: String,
    pub chunk_number: u32,
}

/// Poll for new realtime chunks
pub async fn poll_realtime_chunks(
    site: &str,
    seen: &mut HashSet<String>,
) -> Result<Vec<(String, Vec<u8>)>> {
    let store: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::new()
            .with_bucket_name(REALTIME_BUCKET)
            .with_region("us-east-1")
            .with_skip_signature(true)
            .build()?,
    );

    let prefix = ObjectPath::from(format!("{}/", site));
    let mut list_stream = store.list(Some(&prefix));

    let mut new_chunks = Vec::new();

    while let Some(result) = list_stream.next().await {
        if let Ok(meta) = result {
            let path = meta.location.to_string();
            if !seen.contains(&path) {
                seen.insert(path.clone());

                // Fetch the chunk data
                let object_path = ObjectPath::from(path.clone());
                if let Ok(result) = store.get(&object_path).await {
                    if let Ok(bytes) = result.bytes().await {
                        new_chunks.push((path, bytes.to_vec()));
                    }
                }
            }
        }
    }

    Ok(new_chunks)
}
