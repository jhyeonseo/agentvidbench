"""GCS upload helper for Gemini-family evaluation methods.

`ensure_video_uploaded(local_path, bucket, prefix)` uploads a local video to
gs://<bucket>/<prefix>/<filename> if the blob does not already exist, and
returns the gs:// URI either way. Repeated calls in the same process are
cached so a 100-question run does at most one existence check per unique
video file.

The helper is used by the evaluation orchestrator to translate the local
HF dataset video at ./dataset/videos/videoN.mp4 into a gs:// URI that
Gemini's generate_content / Part.from_uri can consume directly.
"""
from __future__ import annotations

import threading
from pathlib import Path

_uploaded: set[str] = set()
_lock = threading.Lock()


def ensure_video_uploaded(local_path: Path, bucket: str, prefix: str) -> str:
    """Return a gs:// URI for `local_path`, uploading once if needed.

    Args:
        local_path: existing local mp4 file.
        bucket: GCS bucket name (no scheme).
        prefix: object-key prefix; trailing slash is normalized.

    Raises:
        FileNotFoundError if local_path doesn't exist.
        google.api_core.exceptions.* on permission / network failures.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"video not found: {local_path}")

    blob_name = f"{prefix.strip('/')}/{local_path.name}"
    uri = f"gs://{bucket}/{blob_name}"

    with _lock:
        if uri in _uploaded:
            return uri

    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        print(f"  [gcs] uploading {local_path.name} -> {uri}", flush=True)
        blob.upload_from_filename(str(local_path))
    with _lock:
        _uploaded.add(uri)
    return uri
