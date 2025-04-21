import torch
import torch.nn as nn

class MLMDiffMelHarm(nn.Module):
    def __init__(self, 
                 chord_vocab_size,  # V
                 d_model=512, 
                 nhead=8, 
                 num_layers=6, 
                 dim_feedforward=2048,
                 conditioning_dim=16, 
                 device='cuda'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = 513  # 1 + 256 + 256

        # Embedding for condition vector (e.g., style, time sig)
        self.condition_proj = nn.Linear(conditioning_dim, d_model)

        # Melody projection: 140D binary -> d_model
        self.melody_proj = nn.Linear(140, d_model)

        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size)
    # end init

    def forward(self, conditioning_vec, melody_grid, harmony_tokens=None):
        """
        conditioning_vec: (B, C)
        melody_grid: (B, 256, 140)
        harmony_tokens: (B, 256) - optional for training or inference
        """
        B = conditioning_vec.size(0)

        # Project condition: (B, d_model) → (B, 1, d_model)
        cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)

        # Project melody: (B, 256, 140) → (B, 256, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, 256) → (B, 256, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, 256, self.d_model, device=self.device)

        # Concatenate full input: (B, 1 + 256 + 256, d_model)
        full_seq = torch.cat([cond_emb, melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.pos_embedding[:, :self.seq_len, :]

        # Transformer encode
        encoded = self.encoder(full_seq)

        # Optionally decode harmony logits (only last 256 tokens)
        harmony_output = self.output_head(encoded[:, -256:, :])  # (B, 256, V)

        return harmony_output
    # end forward
# end class MLMDiffMelHarm