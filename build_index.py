import os
from models.encoders import MultimodalEncoders
from models.projection import MultimodalProjection
from utils.indexing import FAISSIndex
from tqdm import tqdm

def build_search_index(data_folder='data'):
    """
    Process all media files and build FAISS index
    
    data_folder structure:
        data/
            text/
            images/
            audio/
            videos/
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Initialize components
    encoders = MultimodalEncoders(device)
    projections = MultimodalProjection(device)

    #Load trained projection heads (if available)
    try:
        projections.load_models()
    except:
        print("Warning: Using untrained projection heads")

    index = FAISSIndex(dimension=512)

    #Process each modality
    modalities = {
        'text' : ('text', ['.txt']),
        'images': ('image', ['.jpg', '.jpeg', '.png']),
        'audio' : ('audio', ['.wav', '.mp3']),
        'video' : ('video', ['.mp4', '.avi'])
    }

    for folder, (modality, extensions) in modalities.items():
        folder_path = os.path.join(data_folder, folder)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder} - folder not found")
            continue
        
        files = [f for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in extensions)]
        print(f"\nProcessing {len(files)} {modality} files...")
        
        for filename in tqdm(files):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Encode based on modality
                if modality == 'text':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    embedding = encoders.encode_text(content)
                elif modality == 'image':
                    embedding = encoders.encode_image(file_path)
                elif modality == 'audio':
                    embedding = encoders.encode_audio(file_path)
                elif modality == 'video':
                    embedding = encoders.encode_video(file_path)
                
                # Project to unified space
                unified_embedding = projections.project(embedding, modality)
                
                # Add to index
                metadata = {
                    'path': file_path,
                    'filename': filename,
                    'modality': modality
                }
                index.add_embedding(unified_embedding, metadata)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save index
    index.save()
    print(f"\n✓ Index built successfully with {len(index.metadata)} items!")

if __name__ == "__main__":
    import torch
    build_search_index()