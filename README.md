```markdown
# Multimodal Semantic Search Engine

A cross-modal retrieval system that enables searching across text, images, audio, and video using any modality as input.

## Features
-  Cross-modal search (text→image, image→video, etc.)
-  Fast retrieval using FAISS indexing
-  Interactive Streamlit interface
-  Modular architecture for easy extension

## Setup
1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Add data to `data/` folders
5. Build index: `python build_index.py`
6. Run app: `streamlit run app.py`

## Project Structure
See `docs/architecture.md` for detailed system design.

## Authors
Isha Kale - Int. BTech Computer Science Engineering(AI & Data Science)
MIT WPU, Pune
Sreejit Majumder - Int. BTech Computer Science Engineering(AI & Data Science)
MIT WPU, Pune
Anaya Sharma - Int. BTech Computer Science Engineering(AI & Data Science)
MIT WPU, Pune
Vaibhavi Patil - Int. BTech Computer Science Engineering(AI & Data Science)
MIT WPU, Pune