"""
PictoMusic Configuration Module
Override any setting via environment variables.
"""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = os.getenv("PICTOMUSIC_DATASET_PATH", str(BASE_DIR / "Music.csv"))
EMBEDDINGS_PATH = os.getenv(
    "PICTOMUSIC_EMBEDDINGS_PATH", str(BASE_DIR / "song_embeddings_fp16.npy")
)

CLIP_MODEL_NAME = os.getenv("PICTOMUSIC_CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_BATCH_SIZE = int(os.getenv("PICTOMUSIC_BATCH_SIZE", "32"))
MAX_TOKEN_LENGTH = int(os.getenv("PICTOMUSIC_MAX_TOKEN_LEN", "77"))

DEFAULT_TOP_K = int(os.getenv("PICTOMUSIC_TOP_K", "10"))

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MAX_UPLOAD_SIZE_MB = int(os.getenv("PICTOMUSIC_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
REQUEST_TIMEOUT = int(os.getenv("PICTOMUSIC_REQUEST_TIMEOUT", "10"))
MAX_URL_LENGTH = int(os.getenv("PICTOMUSIC_MAX_URL_LEN", "2048"))
ALLOWED_URL_SCHEMES = ("http", "https")

_BLOCKED_IP_PREFIXES = (
    "127.", "10.", "0.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
    "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.",
    "172.31.", "169.254.", "::1", "fc00:", "fd00:", "fe80:",
)

_IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"RIFF": "webp",
}

SONG_TEXT_COLUMNS = ["name", "artist"]

AUDIO_FEATURE_COLUMNS = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "liveness", "speechiness",
]

# (column_name, low_threshold, high_threshold, low_label, high_label)
FEATURE_DESCRIPTORS = [
    ("danceability",      0.3, 0.7, "not very danceable",   "danceable and groovy"),
    ("energy",            0.3, 0.7, "calm and mellow",      "energetic and intense"),
    ("valence",           0.3, 0.7, "melancholic and dark",  "happy and upbeat"),
    ("acousticness",      0.3, 0.7, "",                      "acoustic"),
    ("instrumentalness",  0.3, 0.7, "",                      "instrumental"),
    ("liveness",          0.3, 0.7, "",                      "live performance"),
    ("speechiness",       0.3, 0.7, "",                      "spoken word"),
]

APP_TITLE = "Pictomusic"
APP_ICON = "🎵"
APP_SUBTITLE = "Upload an image to discover the perfect soundtrack"
APP_VERSION_TAG = "v3.0 Neural Core Active"


def validate_image_url(url: str) -> str:
    """Validate an image URL for safety. Raises ValueError on failure."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()

    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds maximum length of {MAX_URL_LENGTH} characters.")

    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Use one of: {', '.join(ALLOWED_URL_SCHEMES)}"
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname.")

    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        raise ValueError(f"Could not resolve hostname: {hostname}")

    for prefix in _BLOCKED_IP_PREFIXES:
        if ip.startswith(prefix):
            raise ValueError("Access to internal/private network addresses is not allowed.")

    return url


def validate_uploaded_file(file_obj) -> None:
    """Validate an uploaded file (Streamlit UploadedFile). Raises ValueError on failure."""
    if file_obj is None:
        raise ValueError("No file provided.")

    file_obj.seek(0, 2)
    size = file_obj.tell()
    file_obj.seek(0)

    if size > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"File size ({size / (1024*1024):.1f} MB) exceeds the "
            f"maximum of {MAX_UPLOAD_SIZE_MB} MB."
        )

    if size == 0:
        raise ValueError("Uploaded file is empty.")

    name = getattr(file_obj, "name", "")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"File extension '{ext}' is not allowed. "
            f"Use one of: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    header = file_obj.read(12)
    file_obj.seek(0)

    valid = any(header.startswith(magic) for magic in _IMAGE_MAGIC_BYTES)
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        valid = True

    if not valid:
        raise ValueError(
            "File does not appear to be a valid image. "
            "The file header does not match any supported image format."
        )


def escape_html(text: str) -> str:
    """Escape HTML special characters to prevent injection."""
    if not isinstance(text, str):
        return str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
