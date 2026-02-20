import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Tuple, Union
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import faiss
import warnings
import base64
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageMusicRecommender:
    """ image-to-music recommendation system using CLIP embeddings and FAISS."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", embeddings_path: str = "song_embeddings_fp16.npy", dataset_path: str = "Music.csv"):
        """Initialize the recommender with CLIP model, dataset, and FAISS index."""
        self.clip_model_name = clip_model_name
        self.clip_model = None
        self.processor = None
        self.music_df = None
        self.song_description_embeddings_clip = None
        self.index = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings_path = embeddings_path
        self.dataset_path = dataset_path


        self._load_models()
        self._load_dataset()
        self._load_or_generate_embeddings()
        self._build_faiss_index()


    def _load_models(self) -> None:
        """Load CLIP model and processor."""
        try:
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model.eval()
            logging.info(f"✓ CLIP model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"✗ Error loading CLIP model: {e}")
            self.clip_model = None
            self.processor = None


    def _load_dataset(self) -> None:
        """Load the dataset from a  CSV file."""
        try:
            if not os.path.exists(self.dataset_path):
                 logging.error(f"✗ Error: Dataset file not found at {self.dataset_path}. Please ensure 'music.csv' is in the same directory as your script.")
                 self.music_df = None
                 return

            self.music_df = pd.read_csv(self.dataset_path)
            logging.info(f"✓ Dataset loaded: {len(self.music_df)} songs")
        except Exception as e:
            logging.error(f"✗ Error loading dataset: {e}")
            self.music_df = None


    @torch.no_grad()
    def _get_image_embedding(self, image_source: Union[str, Path]) -> Optional[np.ndarray]:
        """Extracting normalized image embeddings using CLIP."""
        if self.clip_model is None or self.processor is None:
            logging.warning("✗ CLIP Model or processor not loaded")
            return None

        try:
            if hasattr(image_source, 'seek') and hasattr(image_source, 'read'):
                 with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_source.name).suffix) as tmp:
                     tmp.write(image_source.getvalue())
                     image_path = tmp.name
            else:
                #  path or URL string
                image_path = image_source


            if str(image_path).startswith('http'):
                response = requests.get(image_path, stream=True, timeout=10)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name
                image = Image.open(tmp_path).convert("RGB")
                os.unlink(tmp_path)
            else:
                image = Image.open(image_path).convert("RGB")


            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

            if hasattr(image_source, 'seek') and hasattr(image_source, 'read'):
                 os.unlink(image_path)

            return features.cpu().numpy()

        except Exception as e:
            logging.error(f"✗ Error processing image: {e}")
            if hasattr(image_source, 'seek') and hasattr(image_source, 'read') and 'image_path' in locals() and os.path.exists(image_path):
                 os.unlink(image_path)
            return None

    @torch.no_grad()
    def _get_text_embeddings(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """Extract normalized text embeddings in batches using CLIP text encoder."""
        if self.clip_model is None or self.processor is None:
            return None

        all_embeddings = []

        logging.info("Starting text embedding generation...")
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt",
                                   padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            embeddings = self.clip_model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())

            if (i // batch_size) % 100 == 0:
                logging.info(f"  Processed batch {i // batch_size} / {total_batches}")


        logging.info("Text embedding generation complete.")
        return torch.cat(all_embeddings, dim=0).numpy()

    def _load_or_generate_embeddings(self):
        """Load song embeddings from file or generate and save them."""
        if self.music_df is None:
            logging.warning("✗ Dataset not loaded. Cannot load or generate embeddings.")
            return

        if os.path.exists(self.embeddings_path):
            try:
                self.song_description_embeddings_clip = np.load(self.embeddings_path)
                logging.info(f"✓ Loaded song embeddings from {self.embeddings_path}")
                if len(self.song_description_embeddings_clip) != len(self.music_df):
                    logging.warning("✗ Warning: Loaded embeddings size mismatch with dataset size. Regenerating embeddings.")
                    self._prepare_song_description_embeddings()
            except Exception as e:
                logging.error(f"✗ Error loading embeddings: {e}. Regenerating embeddings.")
                self._prepare_song_description_embeddings()
        else:
            logging.info(f"No embeddings file found at {self.embeddings_path}. Generating new embeddings...")
            self._prepare_song_description_embeddings()


    def _prepare_song_description_embeddings(self):
        """Generate CLIP text embeddings for song descriptions."""
        if self.clip_model is None or self.processor is None or self.music_df is None:
            logging.warning("✗ CLIP model, processor, or dataset not loaded. Cannot prepare song description embeddings.")
            return

        logging.info("Generating CLIP text embeddings for song descriptions...")

        text_columns = ['name', 'artist']

        missing_columns = [col for col in text_columns if col not in self.music_df.columns]
        if missing_columns:
            logging.error(f"✗ Error: Missing required columns in the dataset: {missing_columns}. Cannot prepare song description embeddings.")
            self.song_description_embeddings_clip = None
            return
        
        logging.info(f"Using columns {text_columns} for text embeddings.")
        song_texts_proxy = self.music_df[text_columns].fillna('').agg(' '.join, axis=1).tolist()

        try:
             self.song_description_embeddings_clip = self._get_text_embeddings(song_texts_proxy)
             logging.info("✓ CLIP text embeddings for song descriptions generated.")
             if self.song_description_embeddings_clip is not None:
                logging.info(f"Shape of CLIP text embeddings: {self.song_description_embeddings_clip.shape}")
                np.save(self.embeddings_path, self.song_description_embeddings_clip)
                logging.info(f"✓ Saved song embeddings to {self.embeddings_path}")

        except Exception as e:
             logging.error(f"✗ Error generating CLIP text embeddings: {e}")
             self.song_description_embeddings_clip = None


    def _build_faiss_index(self):
        """Build a FAISS index for fast similarity search on song description embeddings."""
        if self.song_description_embeddings_clip is None:
            logging.warning("✗ Song description embeddings not available. Cannot build FAISS index.")
            return

        logging.info("Building FAISS index on song description embeddings...")
        try:
            dimension = self.song_description_embeddings_clip.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.song_description_embeddings_clip.astype('float32'))
            logging.info(f"✓ FAISS index built with {self.index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"✗ Error building FAISS index: {e}")
            self.index = None


    def recommend(self, image_source: Union[str, Path],
                 top_k: int = 10,
                 show_scores: bool = True) -> pd.DataFrame:
        """Generate music recommendations based on an input image using FAISS. """
        
        if self.music_df is None or self.index is None or self.clip_model is None or self.processor is None:
            st.error("✗ Recommender not fully initialized. Check logs for errors.")
            return pd.DataFrame()


        img_emb = self._get_image_embedding(image_source)

        if img_emb is None:
            st.error("✗ Failed to process image")
            return pd.DataFrame()

        try:
            distances, indices = self.index.search(img_emb.astype('float32'), top_k)
            similarities = distances[0]

            results = self.music_df.iloc[indices[0]].copy()

            if show_scores:
                results['similarity_score'] = similarities
            return results

        except Exception as e:
            st.error(f"✗ Error during FAISS search: {e}")
            return pd.DataFrame()


    def display_recommendations(self, recommendations: pd.DataFrame) -> None:
        """Display recommendations in a formatted way using Streamlit."""
        if recommendations.empty:
            st.info("No recommendations to display")
            return
        
        cols = ['name', 'artist', 'preview']
        if 'similarity_score' in recommendations.columns:
            cols.append('similarity_score')

        missing_display_cols = [col for col in cols if col not in recommendations.columns and col != 'similarity_score']
        if missing_display_cols:
            st.error(f"✗ Error: Missing display columns in recommendations: {missing_display_cols}. Check your 'music.csv' file.")
            return

        display_df = recommendations[cols].reset_index(drop=True)

        for idx, row in display_df.iterrows():
            with st.container():
                st.markdown(f"### {idx + 1}. {row.get('name', 'N/A')}")
                st.markdown(f"**Artist:** {row.get('artist', 'N/A')}")
                
                if 'similarity_score' in row:
                    st.caption(f"Match Score: {row['similarity_score']:.4f}")
                    
                song_link = row.get('preview', 'N/A')
                if pd.isna(song_link) or song_link == 'no' or not song_link:
                     st.write("🔇 *Preview not available*")
                else:
                     st.audio(song_link, format='audio/mp3')
            st.markdown("---")

# --- UI Styling ---
st.set_page_config(page_title="PictoMusic 🎶", page_icon="🎵", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sleek container for main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Title styling */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Streamlit App Layout ---
st.markdown("<h1>PictoMusic 🎶</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image to discover the perfect soundtrack.</p>", unsafe_allow_html=True)

# Option to upload file or provide URL
image_source_option = st.radio(
    "Choose image source:",
    ("Upload Image", "Image URL")
)

image_source = None

if image_source_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image_source = uploaded_file

elif image_source_option == "Image URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        try:
            st.image(image_url, caption="Image from URL", use_container_width=True)
            image_source = image_url
        except Exception as e:
            st.warning(f"Could not display image from URL. Please check the URL. Error: {e}")
            image_source = None


# Button to trigger recommendation
if st.button("Get Music Recommendations") and image_source is not None:
    @st.cache_resource(show_spinner=False)
    def get_recommender():
        logging.info("Initializing recommender...")
        return ImageMusicRecommender()

    recommender = get_recommender()

    if recommender.music_df is not None and recommender.index is not None and recommender.clip_model is not None and recommender.processor is not None:
        with st.spinner("Analyzing image and finding matching songs... 🎵"):
            recommendations = recommender.recommend(image_source, top_k=10)
        
        if not recommendations.empty:
            st.subheader("Top Recommendations:")
            recommender.display_recommendations(recommendations)
        else:
            st.info("No recommendations found.")
    else:
        st.error("Recommender not initialized properly. Check console logs for errors (like 'music.csv' not found).")