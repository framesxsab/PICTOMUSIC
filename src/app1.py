import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import warnings
import logging

from config import (
    CLIP_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    MAX_TOKEN_LENGTH,
    DEFAULT_TOP_K,
    ALLOWED_IMAGE_EXTENSIONS,
    REQUEST_TIMEOUT,
    DATASET_PATH,
    EMBEDDINGS_PATH,
    SONG_TEXT_COLUMNS,
    FEATURE_DESCRIPTORS,
    APP_TITLE,
    APP_ICON,
    APP_SUBTITLE,
    APP_VERSION_TAG,
    validate_image_url,
    validate_uploaded_file,
    escape_html,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_song_description(row: pd.Series) -> str:
    """Build a rich natural-language description for a song using metadata and audio features."""
    parts: list[str] = []

    name = str(row.get("name", "")).strip()
    artist = str(row.get("artist", "")).strip()
    if name and artist:
        parts.append(f"{name} by {artist}")
    elif name:
        parts.append(name)
    elif artist:
        parts.append(artist)

    descriptors: list[str] = []
    for col, low_thresh, high_thresh, low_label, high_label in FEATURE_DESCRIPTORS:
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        if val <= low_thresh and low_label:
            descriptors.append(low_label)
        elif val >= high_thresh and high_label:
            descriptors.append(high_label)

    if descriptors:
        parts.append(" — " + ", ".join(descriptors))

    return "".join(parts) if parts else "unknown song"


class ImageMusicRecommender:
    """Image-to-music recommendation system using CLIP embeddings and FAISS."""

    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL_NAME,
        embeddings_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ):
        self.clip_model_name = clip_model_name
        self.embeddings_path = embeddings_path or EMBEDDINGS_PATH
        self.dataset_path = dataset_path or DATASET_PATH

        self.clip_model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.music_df: Optional[pd.DataFrame] = None
        self.song_description_embeddings_clip: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_models()
        self._load_dataset()
        self._load_or_generate_embeddings()
        self._build_faiss_index()

    def _load_models(self) -> None:
        try:
            self.clip_model = CLIPModel.from_pretrained(
                self.clip_model_name
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model.eval()
            logging.info("✓ CLIP model loaded on %s", self.device)
        except Exception as e:
            logging.error("✗ Error loading CLIP model: %s", e)
            self.clip_model = None
            self.processor = None

    def _load_dataset(self) -> None:
        try:
            if not os.path.exists(self.dataset_path):
                logging.error("✗ Dataset not found at %s", self.dataset_path)
                self.music_df = None
                return
            self.music_df = pd.read_csv(self.dataset_path)
            logging.info("✓ Dataset loaded: %d songs", len(self.music_df))
        except Exception as e:
            logging.error("✗ Error loading dataset: %s", e)
            self.music_df = None

    @torch.no_grad()
    def _get_image_embedding(self, image_source) -> Optional[np.ndarray]:
        if self.clip_model is None or self.processor is None:
            return None

        tmp_path: Optional[str] = None
        try:
            if hasattr(image_source, "getvalue"):
                suffix = Path(getattr(image_source, "name", "img.jpg")).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(image_source.getvalue())
                    tmp_path = tmp.name
                image = Image.open(tmp_path).convert("RGB")

            elif isinstance(image_source, str) and image_source.startswith("http"):
                validated_url = validate_image_url(image_source)
                response = requests.get(
                    validated_url, stream=True, timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name
                image = Image.open(tmp_path).convert("RGB")

            else:
                image = Image.open(str(image_source)).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().numpy()

        except ValueError as ve:
            logging.warning("Image validation failed: %s", ve)
            return None
        except Exception as e:
            logging.error("✗ Error processing image: %s", e)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @torch.no_grad()
    def _get_text_embeddings(
        self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> Optional[np.ndarray]:
        if self.clip_model is None or self.processor is None:
            return None

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logging.info("Generating text embeddings (%d batches)...", total_batches)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            embeddings = self.clip_model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())

            batch_num = i // batch_size
            if batch_num % 100 == 0:
                logging.info("  Batch %d / %d", batch_num, total_batches)

        return torch.cat(all_embeddings, dim=0).numpy()

    def _load_or_generate_embeddings(self) -> None:
        if self.music_df is None:
            return

        if os.path.exists(self.embeddings_path):
            try:
                self.song_description_embeddings_clip = np.load(self.embeddings_path)
                logging.info("✓ Loaded embeddings from %s", self.embeddings_path)
                if len(self.song_description_embeddings_clip) != len(self.music_df):
                    logging.warning("✗ Embeddings size mismatch — regenerating.")
                    self._prepare_song_description_embeddings()
            except Exception as e:
                logging.error("✗ Error loading embeddings: %s — regenerating.", e)
                self._prepare_song_description_embeddings()
        else:
            logging.info("No embeddings found — generating...")
            self._prepare_song_description_embeddings()

    def _prepare_song_description_embeddings(self) -> None:
        if self.clip_model is None or self.processor is None or self.music_df is None:
            return

        missing = [c for c in SONG_TEXT_COLUMNS if c not in self.music_df.columns]
        if missing:
            logging.error("✗ Missing required columns: %s", missing)
            self.song_description_embeddings_clip = None
            return

        song_texts = self.music_df.apply(build_song_description, axis=1).tolist()

        try:
            self.song_description_embeddings_clip = self._get_text_embeddings(song_texts)
            if self.song_description_embeddings_clip is not None:
                np.save(
                    self.embeddings_path,
                    self.song_description_embeddings_clip.astype(np.float16),
                )
                logging.info("✓ Saved embeddings to %s", self.embeddings_path)
        except Exception as e:
            logging.error("✗ Error generating embeddings: %s", e)
            self.song_description_embeddings_clip = None

    def _build_faiss_index(self) -> None:
        if self.song_description_embeddings_clip is None:
            return

        try:
            dimension = self.song_description_embeddings_clip.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.song_description_embeddings_clip.astype("float32"))
            logging.info("✓ FAISS index built with %d vectors.", self.index.ntotal)
        except Exception as e:
            logging.error("✗ Error building FAISS index: %s", e)
            self.index = None

    @property
    def is_ready(self) -> bool:
        return all([
            self.music_df is not None,
            self.index is not None,
            self.clip_model is not None,
            self.processor is not None,
        ])

    def missing_components(self) -> List[str]:
        missing = []
        if self.music_df is None:
            missing.append("Dataset (Music.csv)")
        if self.index is None:
            missing.append("Embeddings / FAISS index")
        if self.clip_model is None or self.processor is None:
            missing.append("CLIP Model / Processor")
        return missing

    def recommend(
        self,
        image_source,
        top_k: int = DEFAULT_TOP_K,
        show_scores: bool = True,
    ) -> pd.DataFrame:
        if not self.is_ready:
            st.error("✗ Recommender not fully initialized. Check logs.")
            return pd.DataFrame()

        img_emb = self._get_image_embedding(image_source)
        if img_emb is None:
            st.error("✗ Failed to process image. Please try another.")
            return pd.DataFrame()

        try:
            distances, indices = self.index.search(img_emb.astype("float32"), top_k)
            results = self.music_df.iloc[indices[0]].copy()
            if show_scores:
                results["similarity_score"] = distances[0]
            return results
        except Exception as e:
            st.error(f"✗ Error during search: {escape_html(str(e))}")
            return pd.DataFrame()


# =========================================================================
# PREMIUM UI
# =========================================================================

st.set_page_config(
    page_title=f"{APP_TITLE} — AI Music Discovery",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary: #4f06f9;
    --primary-dim: rgba(79, 6, 249, 0.15);
    --accent-pink: #d400ff;
    --accent-blue: #00d4ff;
    --bg-dark: #0a0516;
    --bg-card: rgba(79, 6, 249, 0.06);
    --border-glow: rgba(79, 6, 249, 0.25);
    --text-primary: #f0eef5;
    --text-secondary: #8a85a0;
    --text-muted: #5a5670;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
}

/* Base overrides */
html, body, [data-testid="stAppViewContainer"], .stApp,
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stToolbar"], .stDeployButton {
    display: none !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0820 0%, #0a0516 100%) !important;
    border-right: 1px solid var(--border-glow) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-secondary) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--text-primary) !important;
}

/* Glassmorphism cards */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 1.5rem;
    padding: 2rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 30px rgba(79, 6, 249, 0.08);
}

/* Hero title */
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #4f06f9 0%, #d400ff 50%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 0;
    padding: 0.5rem 0;
}

.hero-subtitle {
    text-align: center;
    color: var(--text-secondary);
    font-size: 1.15rem;
    font-weight: 300;
    margin-top: 0.5rem;
    letter-spacing: 0.01em;
}

.version-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    background: var(--primary-dim);
    border: 1px solid var(--border-glow);
    border-radius: 9999px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--primary);
    margin: 0 auto;
}

.version-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    background: var(--primary);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* Song card */
.song-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 1.25rem;
    padding: 1.5rem;
    margin-bottom: 0.75rem;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.song-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, var(--primary), var(--accent-pink));
    border-radius: 4px 0 0 4px;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.song-card:hover {
    border-color: var(--border-glow);
    transform: translateX(4px);
    box-shadow: 0 8px 32px rgba(79, 6, 249, 0.12);
}

.song-card:hover::before {
    opacity: 1;
}

.song-rank {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 0.25rem;
}

.song-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}

.song-artist {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-top: 0.15rem;
}

/* Match score bar */
.score-container {
    margin-top: 0.75rem;
}

.score-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}

.score-text {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.score-value {
    font-size: 0.75rem;
    font-weight: 800;
    color: var(--primary);
}

.score-bar-bg {
    width: 100%;
    height: 6px;
    background: rgba(79, 6, 249, 0.1);
    border-radius: 9999px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, var(--primary), var(--accent-pink));
    box-shadow: 0 0 12px rgba(79, 6, 249, 0.5);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}

.section-accent {
    color: var(--primary);
}

/* Upload area — nuclear override */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > *,
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > *,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploader"] section > div > div {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-color: transparent !important;
    border: none !important;
}

[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > * {
    background: var(--bg-card) !important;
    background-color: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border-color: var(--border-glow) !important;
}

[data-testid="stFileUploader"] {
    border-radius: 1.25rem !important;
    overflow: hidden !important;
}

[data-testid="stFileUploadDropzone"] {
    border: 2px dashed var(--border-glow) !important;
    border-radius: 1.25rem !important;
    transition: all 0.3s ease !important;
    background-color: rgba(79, 6, 249, 0.04) !important;
    padding: 2.5rem 1.5rem !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 24px rgba(79, 6, 249, 0.15) !important;
    background-color: rgba(79, 6, 249, 0.08) !important;
}

/* Hide default uploader text, replace via CSS */
[data-testid="stFileUploadDropzone"] > div:first-child > div:first-child > span {
    font-size: 0 !important;
}

[data-testid="stFileUploadDropzone"] > div:first-child > div:first-child > span::after {
    content: '🖼️  Drop your image here or browse' !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stFileUploadDropzone"] > div:first-child > div:first-child > small {
    font-size: 0 !important;
}

[data-testid="stFileUploadDropzone"] > div:first-child > div:first-child > small::after {
    content: 'JPG, PNG, WEBP — Max 10 MB' !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] div {
    color: var(--text-secondary) !important;
}

[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background: var(--primary-dim) !important;
    background-color: var(--primary-dim) !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 0.75rem !important;
    font-weight: 600 !important;
}

[data-testid="stBaseButton-secondary"]:hover {
    background-color: rgba(79, 6, 249, 0.25) !important;
    border-color: var(--primary) !important;
}

/* Text input */
.stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 0.75rem !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

.stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 6, 249, 0.2) !important;
}

.stTextInput label {
    color: var(--text-secondary) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(79, 6, 249, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 40px rgba(79, 6, 249, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Radio buttons — pill style */
.stRadio > div {
    background: transparent !important;
    gap: 0.35rem !important;
}

.stRadio > label {
    display: none !important;
}

.stRadio [role="radiogroup"] {
    gap: 0.35rem !important;
}

.stRadio [role="radiogroup"] label {
    background: rgba(79, 6, 249, 0.06) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 0.75rem !important;
    padding: 0.6rem 1rem !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    margin: 0 !important;
}

.stRadio [role="radiogroup"] label:hover {
    background: rgba(79, 6, 249, 0.12) !important;
    border-color: var(--border-glow) !important;
    color: var(--text-primary) !important;
}

.stRadio [role="radiogroup"] label[data-checked="true"],
.stRadio [role="radiogroup"] label:has(input:checked) {
    background: var(--primary-dim) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    box-shadow: 0 0 12px rgba(79, 6, 249, 0.2) !important;
}

.stRadio [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--primary) !important;
}

/* Image display */
[data-testid="stImage"] {
    border-radius: 1.25rem;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
}

/* Audio player */
audio {
    width: 100%;
    height: 40px;
    border-radius: 0.5rem;
    filter: hue-rotate(240deg) saturate(1.5);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-dark);
}
::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-pink);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 0.75rem !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.25rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--primary-dim) !important;
    border-color: var(--border-glow) !important;
    color: var(--primary) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: var(--primary) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 0.75rem !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Divider */
hr {
    border-color: var(--glass-border) !important;
}

/* Stats row */
.stat-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-left: 4px solid var(--primary);
    border-radius: 1rem;
    padding: 1.25rem 1.5rem;
    text-align: left;
}

.stat-label {
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 900;
    color: var(--text-primary);
}

.stat-unit {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary);
    margin-left: 0.35rem;
}

/* Preview unavailable badge */
.no-preview {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--glass-border);
    border-radius: 9999px;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
}

/* Glow background for hero */
.hero-glow {
    position: relative;
}

.hero-glow::before {
    content: '';
    position: absolute;
    top: -100px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(79, 6, 249, 0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: -1;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---- Sidebar ----

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2rem;">
            <div style="width: 42px; height: 42px; background: linear-gradient(135deg, #4f06f9, #6d28d9);
                        border-radius: 12px; display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 0 20px rgba(79,6,249,0.4); font-size: 1.3rem;">
                🎵
            </div>
            <div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #f0eef5; letter-spacing: -0.02em;">
                    Pictomusic
                </div>
                <div style="font-size: 0.6rem; font-weight: 700; color: #4f06f9;
                            letter-spacing: 0.2em; text-transform: uppercase;">
                    AI Audio
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    image_source_option = st.radio(
        "🖼️ Image Source",
        ("Upload Image", "Image URL"),
        label_visibility="visible",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="background: rgba(79,6,249,0.08); border: 1px solid rgba(79,6,249,0.2);
                    border-radius: 1rem; padding: 1rem; margin-top: 1rem;">
            <div style="font-size: 0.6rem; font-weight: 700; color: #5a5670;
                        letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;">
                Neural Engine
            </div>
            <div style="font-size: 0.8rem; color: #8a85a0; line-height: 1.5;">
                Powered by <span style="color: #4f06f9; font-weight: 700;">CLIP</span> vision-language
                model + <span style="color: #4f06f9; font-weight: 700;">FAISS</span> similarity search
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- Main Content ----

# Hero section
st.markdown(
    f"""
    <div style="text-align: center; padding: 1rem 0 0.5rem;" class="hero-glow">
        <div class="version-badge">{escape_html(APP_VERSION_TAG)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<h1 class="hero-title">WHERE SOUND<br>BECOMES SIGHT</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">{escape_html(APP_SUBTITLE)}</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Image input
image_source = None
col_pad_l, col_main, col_pad_r = st.columns([1, 3, 1])

with col_main:
    if image_source_option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Drop your image here",
            type=[ext.lstrip(".") for ext in ALLOWED_IMAGE_EXTENSIONS],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            try:
                validate_uploaded_file(uploaded_file)
                st.image(uploaded_file, use_container_width=True)
                image_source = uploaded_file
            except ValueError as ve:
                st.error(f"⚠️ {ve}")

    elif image_source_option == "Image URL":
        image_url = st.text_input(
            "Enter Image URL",
            placeholder="https://example.com/image.jpg",
            label_visibility="collapsed",
        )
        if image_url:
            try:
                validated_url = validate_image_url(image_url)
                st.image(validated_url, use_container_width=True)
                image_source = validated_url
            except ValueError as ve:
                st.error(f"⚠️ {ve}")
            except Exception as e:
                st.warning(f"Could not load image. Error: {escape_html(str(e))}")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  ANALYZE & DISCOVER", use_container_width=True):
        if image_source is not None:
            @st.cache_resource(show_spinner=False)
            def get_recommender():
                return ImageMusicRecommender()

            recommender = get_recommender()

            if recommender.is_ready:
                with st.spinner("Neural engine analyzing visual frequencies..."):
                    recommendations = recommender.recommend(image_source)

                if not recommendations.empty:
                    st.session_state["recommendations"] = recommendations
                    st.session_state["show_results"] = True
                else:
                    st.info("No matching frequencies found. Try a different image.")
            else:
                st.error(
                    "Neural engine offline. Missing: "
                    + ", ".join(recommender.missing_components())
                )
        else:
            st.warning("Please provide an image first.")


# ---- Results ----

if st.session_state.get("show_results") and "recommendations" in st.session_state:
    recommendations = st.session_state["recommendations"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats row
    if "similarity_score" in recommendations.columns:
        top_score = recommendations["similarity_score"].max()
        avg_score = recommendations["similarity_score"].mean()
        num_results = len(recommendations)

        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Top Match</div>
                    <div class="stat-value">{top_score:.3f}<span class="stat-unit">SIM</span></div>
                </div>
                """, unsafe_allow_html=True,
            )
        with stat_cols[1]:
            st.markdown(
                f"""
                <div class="stat-card" style="border-left-color: #d400ff;">
                    <div class="stat-label">Avg Score</div>
                    <div class="stat-value">{avg_score:.3f}<span class="stat-unit" style="color:#d400ff;">AVG</span></div>
                </div>
                """, unsafe_allow_html=True,
            )
        with stat_cols[2]:
            st.markdown(
                f"""
                <div class="stat-card" style="border-left-color: #00d4ff;">
                    <div class="stat-label">Tracks Found</div>
                    <div class="stat-value">{num_results}<span class="stat-unit" style="color:#00d4ff;">TRACKS</span></div>
                </div>
                """, unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">🎵 Neural <span class="section-accent">Recommendations</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.5rem;">'
        'Tracks ranked by visual-audio resonance</p>',
        unsafe_allow_html=True,
    )

    display_cols = ["name", "artist", "preview"]
    if "similarity_score" in recommendations.columns:
        display_cols.append("similarity_score")

    missing_cols = [
        c for c in display_cols
        if c not in recommendations.columns and c != "similarity_score"
    ]
    if missing_cols:
        st.error(f"✗ Missing columns: {', '.join(missing_cols)}. Check Music.csv.")
    else:
        max_score = recommendations["similarity_score"].max() if "similarity_score" in recommendations.columns else 1.0

        for idx, row in recommendations[display_cols].reset_index(drop=True).iterrows():
            song_name = escape_html(str(row.get("name", "N/A")))
            artist_name = escape_html(str(row.get("artist", "N/A")))
            score = row.get("similarity_score", 0)
            score_pct = min((score / max_score) * 100, 100) if max_score > 0 else 0

            st.markdown(
                f"""
                <div class="song-card">
                    <div class="song-rank">Track #{idx + 1}</div>
                    <div class="song-name">{song_name}</div>
                    <div class="song-artist">{artist_name}</div>
                    <div class="score-container">
                        <div class="score-label">
                            <span class="score-text">Neural Match</span>
                            <span class="score-value">{score:.4f}</span>
                        </div>
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width: {score_pct:.1f}%;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            song_link = row.get("preview", "")
            if pd.isna(song_link) or not song_link or str(song_link).strip().lower() == "no":
                st.markdown(
                    '<div class="no-preview">🔇 Preview not available</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.audio(str(song_link), format="audio/mp3")

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)