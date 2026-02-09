import streamlit as st
import torch
from models.encoders import MultimodalEncoders
from models.projection import MultimodalProjection
from utils.indexing import FAISSIndex
from PIL import Image
import os

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.encoders = None
    st.session_state.projections = None
    st.session_state.index = None

@st.cache_resource
def load_models():
    """Load all models once"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoders = MultimodalEncoders(device)
    projections = MultimodalProjection(device)
    
    try:
        projections.load_models()
    except:
        st.warning("Projection heads not trained yet")
    
    index = FAISSIndex()
    try:
        index.load()
    except:
        st.error("Index not found! Run build_index.py first")
        return None, None, None
    
    return encoders, projections, index

def search_query(query, query_type, encoders, projections, index, top_k=10):
    """Process query and return results"""
    
    # Encode query
    if query_type == 'text':
        embedding = encoders.encode_text(query)
    elif query_type == 'image':
        embedding = encoders.encode_image(query)
    elif query_type == 'audio':
        embedding = encoders.encode_audio(query)
    elif query_type == 'video':
        embedding = encoders.encode_video(query)
    
    # Project to unified space
    unified_embedding = projections.project(embedding, query_type)
    
    # Search
    results = index.search(unified_embedding, k=top_k)
    
    return results

def main():
    st.set_page_config(page_title="Multimodal Search Engine", layout="wide")
    
    st.title("🔍 Multimodal Semantic Search Engine")
    st.markdown("Search across text, images, audio, and video using any modality!")
    
    # Load models
    if not st.session_state.initialized:
        with st.spinner("Loading models... This may take a minute"):
            encoders, projections, index = load_models()
            if encoders is None:
                st.stop()
            
            st.session_state.encoders = encoders
            st.session_state.projections = projections
            st.session_state.index = index
            st.session_state.initialized = True
    
    # Query interface
    st.sidebar.header("Query Input")
    query_mode = st.sidebar.selectbox(
        "Select Query Type",
        ["Text", "Image", "Audio", "Video"]
    )
    
    query = None
    query_type = query_mode.lower()
    
    if query_mode == "Text":
        query = st.sidebar.text_area("Enter your search query:", height=100)
    else:
        uploaded_file = st.sidebar.file_uploader(
            f"Upload {query_mode}",
            type=['jpg', 'png', 'jpeg'] if query_mode == 'Image' else
                 ['wav', 'mp3'] if query_mode == 'Audio' else
                 ['mp4', 'avi']
        )
        
        if uploaded_file:
            # Save temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            query = temp_path
    
    top_k = st.sidebar.slider("Number of results", 5, 20, 10)
    
    # Search button
    if st.sidebar.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = search_query(
                    query, 
                    query_type,
                    st.session_state.encoders,
                    st.session_state.projections,
                    st.session_state.index,
                    top_k
                )
            
            # Display results
            st.header("Search Results")
            
            cols = st.columns(3)
            for idx, result in enumerate(results):
                col = cols[idx % 3]
                
                with col:
                    st.subheader(f"#{idx+1} - Score: {result['score']:.3f}")
                    
                    metadata = result['metadata']
                    modality = metadata['modality']
                    file_path = metadata['path']
                    
                    # Display based on modality
                    if modality == 'image':
                        st.image(file_path, use_column_width=True)
                    elif modality == 'text':
                        with open(file_path, 'r') as f:
                            st.text(f.read()[:200] + "...")
                    elif modality == 'audio':
                        st.audio(file_path)
                    elif modality == 'video':
                        st.video(file_path)
                    
                    st.caption(f"**Type:** {modality} | **File:** {metadata['filename']}")
            
            # Cleanup temp file
            if query_type != 'text' and os.path.exists(query):
                os.remove(query)
        else:
            st.warning("Please provide a query!")

if __name__ == "__main__":
    main()


## Part 4: Running Your System

### Step-by-Step Execution
"""
**1. Prepare Your Data**

Create this structure:
```
data/
  ├── text/       (put .txt files here)
  ├── images/     (put .jpg, .png files)
  ├── audio/      (put .wav, .mp3 files)
  └── videos/     (put .mp4 files)
"""
