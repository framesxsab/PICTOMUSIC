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
    UI_BG_COLOR,
    UI_TEXT_COLOR,
    UI_SUBTITLE_COLOR,
    UI_GRADIENT_START,
    UI_GRADIENT_END,
    UI_FONT_FAMILY,
    UI_MAX_WIDTH,
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
        """Load CLIP model and processor."""
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
        """Load the music dataset from CSV."""
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
    def _get_image_embedding(
        self, image_source: Union[str, Path, object]
    ) -> Optional[np.ndarray]:
        """Extract normalized image embedding using CLIP."""
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
        """Extract normalized text embeddings in batches using CLIP."""
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

        logging.info("Text embedding generation complete.")
        return torch.cat(all_embeddings, dim=0).numpy()

    def _load_or_generate_embeddings(self) -> None:
        """Load pre-computed song embeddings or generate fresh ones."""
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
        """Generate CLIP text embeddings for rich song descriptions."""
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
                logging.info("✓ Embeddings shape: %s", self.song_description_embeddings_clip.shape)
                np.save(
                    self.embeddings_path,
                    self.song_description_embeddings_clip.astype(np.float16),
                )
                logging.info("✓ Saved embeddings to %s", self.embeddings_path)
        except Exception as e:
            logging.error("✗ Error generating embeddings: %s", e)
            self.song_description_embeddings_clip = None

    def _build_faiss_index(self) -> None:
        """Build FAISS index for fast similarity search."""
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
        image_source: Union[str, Path],
        top_k: int = DEFAULT_TOP_K,
        show_scores: bool = True,
    ) -> pd.DataFrame:
        """Generate music recommendations for an input image."""
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

    def display_recommendations(self, recommendations: pd.DataFrame) -> None:
        """Render recommendation cards in Streamlit."""
        if recommendations.empty:
            st.info("No recommendations to display.")
            return

        display_cols = ["name", "artist", "preview"]
        if "similarity_score" in recommendations.columns:
            display_cols.append("similarity_score")

        missing_cols = [
            c for c in display_cols
            if c not in recommendations.columns and c != "similarity_score"
        ]
        if missing_cols:
            st.error(f"✗ Missing columns: {', '.join(missing_cols)}. Check Music.csv.")
            return

        for idx, row in recommendations[display_cols].reset_index(drop=True).iterrows():
            with st.container():
                song_name = escape_html(str(row.get("name", "N/A")))
                artist_name = escape_html(str(row.get("artist", "N/A")))

                st.markdown(f"### {idx + 1}. {song_name}")
                st.markdown(f"**Artist:** {artist_name}")

                if "similarity_score" in row:
                    st.caption(f"Match Score: {row['similarity_score']:.4f}")

                song_link = row.get("preview", "")
                if pd.isna(song_link) or not song_link or str(song_link).strip().lower() == "no":
                    st.write("🔇 *Preview not available*")
                else:
                    st.audio(str(song_link), format="audio/mp3")
            st.markdown("---")


st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="centered")

st.markdown(
    f"""
    <style>
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stApp {{
            background-color: {UI_BG_COLOR};
            color: {UI_TEXT_COLOR};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: {UI_MAX_WIDTH};
        }}
        h1 {{
            font-family: {UI_FONT_FAMILY};
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, {UI_GRADIENT_START}, {UI_GRADIENT_END});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            text-align: center;
            color: {UI_SUBTITLE_COLOR};
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<h1>{escape_html(APP_TITLE)}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{escape_html(APP_SUBTITLE)}</p>", unsafe_allow_html=True)

image_source_option = st.radio("Choose image source:", ("Upload Image", "Image URL"))
image_source = None

if image_source_option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=[ext.lstrip(".") for ext in ALLOWED_IMAGE_EXTENSIONS],
    )
    if uploaded_file is not None:
        try:
            validate_uploaded_file(uploaded_file)
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            image_source = uploaded_file
        except ValueError as ve:
            st.error(f"⚠️ {ve}")

elif image_source_option == "Image URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        try:
            validated_url = validate_image_url(image_url)
            st.image(validated_url, caption="Image from URL", use_container_width=True)
            image_source = validated_url
        except ValueError as ve:
            st.error(f"⚠️ {ve}")
        except Exception as e:
            st.warning(f"Could not display image from URL. Error: {escape_html(str(e))}")

if st.button("Get Music Recommendations") and image_source is not None:
    @st.cache_resource(show_spinner=False)
    def get_recommender():
        return ImageMusicRecommender()

    recommender = get_recommender()

    if recommender.is_ready:
        with st.spinner("Analyzing image and finding matching songs... 🎵"):
            recommendations = recommender.recommend(image_source)

        if not recommendations.empty:
            st.subheader("Top Recommendations:")
            recommender.display_recommendations(recommendations)
        else:
            st.info("No recommendations found.")
    else:
        st.error(
            "Recommender failed to start. Missing: "
            + ", ".join(recommender.missing_components())
            + ". Check the app logs."
        )