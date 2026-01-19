//! S3 access for NEXRAD data (internal)

pub mod archive;
pub mod realtime;

pub use archive::fetch_archive_file;
pub use realtime::poll_realtime_chunks;
