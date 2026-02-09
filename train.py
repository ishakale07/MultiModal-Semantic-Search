import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.encoders import MultimodalEncoders
from models.projection import MultimodalProjection

class ContrastiveLoss(nn.Module):
    """Contrastive loss to align modalities"""

    def __init__(self, temperature=0.07):
        super().__init()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings1, embeddings2):
        #compute similarity matrix
        embeddings1 = nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = nn.functional.normalize(embeddings2, p=2, dim=1)

        similarity_matrix = torch.matmul(embeddings1,embeddings2.T)/self.temperature

        #labels - diagonal elements are positive pairs
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

        #compute loss in both directions
        loss1 =  self.criterion(similarity_matrix, labels)
        loss2 = self.criterion(similarity_matrix.T, labels)

        return (loss1 + loss2) / 2
    
def train_projection_heads(dataset_path, num_epochs=30, batch_size=32):
    """
    Train projection heads using contrastive learning
    
    dataset_path: folder with paired data (e.g., image-text pairs)
    """
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #initialize models
    encoders = MultimodalEncoders(device)
    projections = MultimodalProjection(device)

    #loss and optimizer
    criterion = ContrastiveLoss(temperature=0.07)

    #collect all parameters from projection heads
    params = []
    for proj in projections.projections.values():
        params.extend(list(proj.parameters()))

    optimizer = optim.AdamW(params, lr=1e-4, weight_decay =1e-4)

    #Training loop
    for epoch in range(num_epochs):
        total_loss = 0

        # Here you would load your paired data
        # For demo purposes, showing the structure:
        """
        for batch in dataloader:
            text_emb = encoders.encode_text(batch['text'])
            image_emb = encoders.encode_image(batch['image'])
            
            # Project to unified space
            text_proj = projections.project(text_emb, 'text')
            image_proj = projections.project(image_emb, 'image')
            
            # Compute loss
            loss = criterion(text_proj, image_proj)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        """
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    
    # Save trained models
    projections.save_models()
    print("Training complete!")

if __name__ == "__main__":
    # You'll need to prepare your dataset first
    print("Note: You need paired multimodal data for training")
    print("Example datasets: MS-COCO, Conceptual Captions, AudioCaps")
    