import torch
import torch.nn as nn

class GridMLMMelHarm(nn.Module):
    def __init__(self, 
                 chord_vocab_size,  # V
                 d_model=512, 
                 nhead=8, 
                 num_layers=8, 
                 dim_feedforward=2048,
                 conditioning_dim=16,
                 pianoroll_dim=100,
                 grid_length=256,
                 dropout=0.3,
                 max_stages=10,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = 1 + grid_length + grid_length # condition + melody + harmony
        self.grid_length = grid_length
        # Embedding for condition vector (e.g., style, time sig)
        self.condition_proj = nn.Linear(conditioning_dim, d_model, device=self.device)
        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # embedding for curriculum stage
        self.max_stages = max_stages
        self.stage_embedding_dim = 64
        self.stage_embedding = nn.Embedding(self.max_stages, self.stage_embedding_dim, device=self.device)
        # New projection layer to go from (d_model + stage_embedding_dim) → d_model
        self.stage_proj = nn.Linear(self.d_model + self.stage_embedding_dim, self.d_model, device=self.device)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(self, conditioning_vec, melody_grid, harmony_tokens=None, stage_indices=None):
        """
        conditioning_vec: (B, C)
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = conditioning_vec.size(0)

        # Project condition: (B, d_model) → (B, 1, d_model)
        cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=self.device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([cond_emb, melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.pos_embedding[:, :self.seq_len, :]
        if stage_indices is not None:
            stage_emb = self.stage_embedding(stage_indices)  # (B, stage_embedding_dim)
            stage_emb = stage_emb.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, stage_embedding_dim)
            # Concatenate along the feature dimension
            full_seq = torch.cat([full_seq, stage_emb], dim=-1)  # (B, seq_len, d_model + stage_embedding_dim)
            # Project back to d_model
            full_seq = self.stage_proj(full_seq)  # (B, seq_len, d_model)

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # Transformer encode
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)

        return harmony_output
    # end forward
# end class GridMLMMelHarm

class GridMLMMelHarmNoStage(nn.Module):
    def __init__(self, 
                 chord_vocab_size,  # V
                 d_model=512, 
                 nhead=8, 
                 num_layers=8, 
                 dim_feedforward=2048,
                 conditioning_dim=16,
                 pianoroll_dim=100,
                 grid_length=256,
                 dropout=0.3,
                 max_stages=10,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = 1 + grid_length + grid_length # condition + melody + harmony
        self.grid_length = grid_length
        # Embedding for condition vector (e.g., style, time sig)
        self.condition_proj = nn.Linear(conditioning_dim, d_model, device=self.device)
        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # # embedding for curriculum stage
        # self.max_stages = max_stages
        # self.stage_embedding_dim = 64
        # self.stage_embedding = nn.Embedding(self.max_stages, self.stage_embedding_dim, device=self.device)
        # # New projection layer to go from (d_model + stage_embedding_dim) → d_model
        # self.stage_proj = nn.Linear(self.d_model + self.stage_embedding_dim, self.d_model, device=self.device)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(self, conditioning_vec, melody_grid, harmony_tokens=None, stage_indices=None):
        """
        conditioning_vec: (B, C)
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = conditioning_vec.size(0)

        # Project condition: (B, d_model) → (B, 1, d_model)
        cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=self.device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([cond_emb, melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.pos_embedding[:, :self.seq_len, :]
        # if stage_indices is not None:
        #     stage_emb = self.stage_embedding(stage_indices)  # (B, stage_embedding_dim)
        #     stage_emb = stage_emb.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, stage_embedding_dim)
        #     # Concatenate along the feature dimension
        #     full_seq = torch.cat([full_seq, stage_emb], dim=-1)  # (B, seq_len, d_model + stage_embedding_dim)
        #     # Project back to d_model
        #     full_seq = self.stage_proj(full_seq)  # (B, seq_len, d_model)

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # Transformer encode
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)

        return harmony_output
    # end forward
# end class GridMLMMelHarmNoStage