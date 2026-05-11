//! Realtime data access from S3

use crate::error::Result;
use crate::fetch::{FETCH_SEMAPHORE, store_for_bucket};
use futures::stream::StreamExt;
use object_store::{ObjectStore, ObjectStoreExt};
use object_store::path::Path as ObjectPath;
use std::collections::HashSet;

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
    let store = store_for_bucket(REALTIME_BUCKET)?;

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
                let _permit = FETCH_SEMAPHORE.acquire().await.expect("semaphore closed");
                if let Ok(result) = store.get(&object_path).await
                    && let Ok(bytes) = result.bytes().await {
                        new_chunks.push((path, bytes.to_vec()));
                    }
            }
        }
    }

    Ok(new_chunks)
}
