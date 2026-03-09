---
title: Pictomusic
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Pictomusic — AI Music Discovery

### 🌌 Where Sight Becomes Sound

Upload an image and discover the perfect soundtrack. Powered by **CLIP** vision-language model and **FAISS** similarity search.

🚀 **[Try Pictomusic live on Hugging Face Spaces!](https://huggingface.co/spaces/fxsab/pictomusic)** 🚀

<br>

---

## 🎵 What is Pictomusic?

Pictomusic is an AI-powered application that bridges the gap between visual and audio media. By uploading an image (or providing an image URL), the application's Neural Engine analyzes the visual mood, themes, and elements of the image, and then cross-references them against a massive database of songs to recommend the perfect accompanying soundtrack.

## 🛠️ Technology Stack & Architecture

Pictomusic relies on a seamless integration of modern AI models and a fast, responsive backend:

### AI & Machine Learning
- **OpenAI CLIP (`clip-vit-base-patch32`)**: The core of the Neural Engine. CLIP encodes both the uploaded image and the descriptive text of over 65,000 songs into a shared mathematical vector space. 
- **FAISS (Facebook AI Similarity Search)**: Used to index the pre-computed text embeddings of the song dataset. When an image is uploaded, FAISS rapidly calculates the closest `<Image Vector, Text Vector>` pairs using Inner Product similarity to return the best-matching songs.
- **PyTorch & Transformers**: Used to handle the inference of the CLIP models on CPU efficiently.

### Backend & Data Processing
- **Python 3.11**: The core programming language.
- **Pandas & NumPy**: For efficient dataset manipulation, feature extraction, and handling the FP16 `song_embeddings_fp16.npy` vectors.
- **Scikit-Learn**: Used during the data preprocessing pipelines.

### Frontend
- **Streamlit**: Provides the rapid, interactive web application UI.
- **Custom CSS / Glassmorphism**: The UI was completely restyled using injected CSS to feature a premium dark-mode aesthetic, vibrant gradients, custom scrollbars, and interactive hover states.

### Deployment & DevOps
- **Docker**: The application is containerized using a custom `Dockerfile` based on `python:3.11-slim`.
- **Hugging Face Spaces**: Hosted on HF Spaces using the Docker SDK. The deployment is specifically tuned to run as a non-root user (UID 1000) with disabled XSRF/CORS Streamlit protections to ensure smooth iframe embedding.
- **Git LFS**: Used to track the heavy `song_embeddings_fp16.npy` file so the FAISS index doesn't have to be generated on the fly.

---

## ⚡ Features

- 🖼️ **Flexible Inputs:** Support for both direct file uploads (JPG, PNG, WEBP) and Image URLs.
- 🧠 **Neural Matching:** Real-time visual-audio resonance calculation, returning a Similarity Score (SIM) for how strongly a song matches an image.
- 🎧 **Audio Previews:** Listen to 30-second previews of the recommended tracks directly within the app interface.
- 📱 **Responsive UI:** A stunning, desktop-and-mobile friendly UI built with Custom HTML/CSS over Streamlit.
- 🚀 **Lightning Fast:** By pre-computing 65,000+ song embeddings into an FP16 NumPy array, search results are returned in milliseconds via FAISS.

---

## 🖥️ Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/framesxsab/tcb-project.git
   cd tcb-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Data Files exist:
   You will need the `Music.csv` and `song_embeddings_fp16.npy` files in the root directory.

4. Run the app:
   ```bash
   streamlit run src/app1.py
   ```
