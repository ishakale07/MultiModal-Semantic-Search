import torch 
import torch.nn as nn

class ProjectionHead(nn.Module):
    """Maps encoder outputs to unified 512-dim space"""

    def __init__(self, input_dim, output_dim=512, hidden_dim=1024):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)
    
class MultimodalProjection:
    """Manages all projection heads"""

    def __init__(self, device='cpu'):
        self.device = device

        # Different input dimensions for each modality
        self.text_proj = ProjectionHead(768, 512).to(device) # BERT output
        self.image_proj = ProjectionHead(512, 512).to(device)  # CLIP output
        self.audio_proj = ProjectionHead(512, 512).to(device)  # Whisper output
        self.video_proj = ProjectionHead(512, 512).to(device)  # Video (CLIP-based)

        self.projections = {
            'text' : self.text_proj,
            'image' : self.image_proj,
            'audio': self.audio_proj,
            'video': self.video_proj
        }

    def project(self, embedding, modality):
        """Project embedding to unified space"""
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            projected = self.projections[modality](embedding_tensor)
            # L2 normalize for cosine similarity
            projected = torch.nn.functional.normalize(projected, p=2, dim=1)

        return projected.cpu().numpy().squeeze()
    
    def save_models(self, save_dir='models/checkpoints'):
        """Save trained projection heads"""
        import os

        for name, model in self.projections.items():
            torch.save(model.state_dict(), f'{save_dir}/{name}_projection.pth')
        print(f"Models saved to {save_dir}")

    def load_models(self, save_dir='models/checkpoints'):
        """Load trained projection heads"""
        for name, model in self.projections.items():
            model.load_state_dict(torch.load(f'{save_dir}/{name}_projection.pth'))
            model.eval()
        print(f"Models loaded from {save_dir}")