import torch
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor
import whisper
import cv2
import librosa
import numpy as np
from PIL import Image

class MultimodalEncoders:
    """Handles encoding of all 4 modalities"""

    def __init__(self, device='cpu'):
        self.device=device
        print("Loading models... This may take a few minutes.")

        # Text Encoder (BERT)
        print("Loading BERT...")
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.text_encoder.eval()

        # Image Encoder (CLIP)
        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.image_encoder = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        self.image_encoder.eval()

        # Audio Encoder (Whisper)
        print("Loading Whisper...")
        self.audio_encoder = whisper.load_model("base").to(device)

        print("All models loaded successfully!")

    def encode_text(self, text):
        """Convert text to 768-dim embedding"""
        inputs = self.text_tokenizer(
            text,
            return_tensors='pt',
            padding = True,
            truncation = True,
            max_length = 123
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.squeeze()
    
    def encode_image(self, image_path):
        """Convert image to 512-dim embedding"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(images=image, return_tensors='pt').to(self.device)

        with torch.no_grad():
            embedding = self.image_encoder.get_image_features(**inputs).cpu().numpy()
        
        return embedding.squeeze()
    
    def encode_audio(self, audio_path):
        """Convert image to 512-dim embedding"""
        # Load and resample to 16kHz (Whisper requirement)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Get Whisper encoder output
        with torch.no_grad():
            #convert to log-mel spectrogram
            mel = whisper.log_mel_spectrogram(torch.tensor(audio)).to(self.device)
            # get encoder features
            embedding = self.audio_encoder.encoder(mel.unsqueeze(0))
            embedding = embedding.mean(dim=1).cpu().numpy()

        return embedding.squeeze()
    
    def encode_video(self, video_path, num_frames=16):
        """Convert video to embedding by sampling frames"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frame_embeddings = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtCOlor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                #Encode frame using CLIP
                inputs = self.image_processor(images = image, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    embedding = self.image_encoder.get_image_features(**inputs).cpu().numpy()
                frame_embeddings.append(embedding)

        cap.release()

        #Average frame embeddings
        video_embedding = np.mean(frame_embeddings, axis=0)
        return video_embedding.squeeze()