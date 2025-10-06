import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import math
from copy import deepcopy

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


def sinusoidal_positional_encoding(seq_len, d_model, device):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)
# end sinusoidal_positional_encoding

# ========== DUAL ENCODER MODEL ==========

# ========== Small helper: Transformer encoder layer with cross-attention ==========
class HarmonyEncoderLayerWithCross(nn.Module):
    """
    One layer for harmony encoder:
      - self-attn (harmony -> harmony)
      - cross-attn (harmony queries -> melody keys/values)
      - feed-forward
    Stores last attention weights for diagnostics.
    """
    def __init__(
                self,
                d_model, 
                nhead, 
                dim_feedforward=2048, 
                dropout=0.3, 
                activation='gelu', 
                batch_first=True,
                device='cpu'
            ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device)

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device)

        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.norm3 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # placeholders for attention visualization
        self.last_self_attn = None  # shape (B, nhead, Lh, Lh) if requested
        self.last_cross_attn = None # shape (B, nhead, Lh, Lm) if requested
    # end init

    def forward(self, x_h, melody_kv, attn_mask=None, key_padding_mask=None, melody_key_padding_mask=None):
        """
        x_h: (B, Lh, d_model) harmony input
        melody_kv: (B, Lm, d_model) melody encoded (keys & values)
        attn_mask: optional for self-attn
        key_padding_mask: optional for self-attn
        melody_key_padding_mask: optional for cross-attn (for melody padding)
        """
        # Self-attention
        h2, self_w = self.self_attn(x_h, x_h, x_h,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        # self_w : (B, nhead, Lh, Lh)  if batch_first and average_attn_weights=False
        self.last_self_attn = self_w.detach() if isinstance(self_w, torch.Tensor) else None

        x_h = x_h + self.dropout1(h2)
        x_h = self.norm1(x_h)
        # x_h = self.norm1(h2)

        # Cross-attention: queries = harmony (x_h), keys/values = melody_kv
        c2, cross_w = self.cross_attn(x_h, melody_kv, melody_kv,
                                      key_padding_mask=melody_key_padding_mask,
                                      need_weights=True,
                                      average_attn_weights=False)
        self.last_cross_attn = cross_w.detach() if isinstance(cross_w, torch.Tensor) else None

        x_h = x_h + self.dropout2(c2)
        x_h = self.norm2(x_h)
        # x_h = self.norm2(c2)

        # Feed-forward
        ff = self.linear2(self.dropout3(self.activation(self.linear1(x_h))))
        x_h = x_h + self.dropout3(ff)
        x_h = self.norm3(x_h)

        return x_h
    # end forward
# end class HarmonyEncoderLayerWithCross

# ========== Stacked encoder modules ==========
class SimpleTransformerStack(nn.Module):
    """A small wrapper that stacks standard nn.TransformerEncoderLayer layers (no cross-attn)."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
    # end init

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, L, D) assumed batch_first=True
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
    # end forward
# end class SimpleTransformerStack

class HarmonyTransformerStack(nn.Module):
    """Stack of HarmonyEncoderLayerWithCross"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
    # end init

    def forward(self, x_h, melody_kv, h_key_padding_mask=None, melody_key_padding_mask=None):
        # x_h: (B, Lh, D)
        for layer in self.layers:
            x_h = layer(x_h, melody_kv,
                        key_padding_mask=h_key_padding_mask,
                        melody_key_padding_mask=melody_key_padding_mask)
        return x_h
    # end forward
# end class HarmonyTransformerStack

# ========== Dual-encoder model ==========
class DualGridMLMMelHarm(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=8,
                 num_layers_harm=8,
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 melody_length=80,
                 harmony_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)   # project PCP (and bar flag) -> d_model
        # self.melody_proj = nn.Embedding(chord_vocab_size, d_model, device=device)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            max(melody_length, harmony_length), d_model, device
        )

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = nn.TransformerEncoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       device=device)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerWithCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', 
                                                  batch_first=True, device=device)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)

        # Norms / dropout
        self.input_norm = nn.LayerNorm(d_model, device=device)
        self.output_norm = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

        self.to(device)
    # end init

    def forward(self, melody_grid, harmony_tokens=None,
                melody_key_padding_mask=None, harm_key_padding_mask=None):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        stage_indices: (B,) or (B,1) ints in [0, max_stages-1] - used by harmony encoder
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = self.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.shared_pos[:, :self.melody_length, :]
        mel = self.input_norm(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=melody_key_padding_mask)  # (B, Lm, d_model)

        # ---- Harmony embedding + optional stage conditioning ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.shared_pos[:, :self.harmony_length, :]

        harm = self.input_norm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded,
                                            h_key_padding_mask=harm_key_padding_mask,
                                            melody_key_padding_mask=melody_key_padding_mask)  # (B, Lh, d_model)

        harm_encoded = self.output_norm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            self_attns.append(layer.last_self_attn)
            cross_attns.append(layer.last_cross_attn)
        return self_attns, cross_attns
    # end get_attention_maps
# end class DualGridMLMMelHarm

# ========== SINGLE ENCODER MODEL ==========

class TransformerEncoderLayerWithAttn(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # place to store the weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # same as parent forward, except we intercept attn_weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn_weights = attn_weights.detach()  # store for later

        # rest of the computation is copied from TransformerEncoderLayer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
# end TransformerEncoderLayerWithAttn

class SingleGridMLMelHarm(nn.Module):
    def __init__(self, 
                 chord_vocab_size,  # V
                 d_model=512, 
                 nhead=4, 
                 num_layers=4, 
                 dim_feedforward=2048,
                 pianoroll_dim=13,      # PCP + bars only
                 grid_length=80,
                 dropout=0.3,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = 1 + grid_length + grid_length # condition + melody + harmony
        self.grid_length = grid_length

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        # # Positional embeddings (separate for clarity)
        self.shared_pos = sinusoidal_positional_encoding(
            grid_length, d_model, device
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
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

    def forward(self, melody_grid, harmony_tokens=None):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = melody_grid.size(0)
        device = self.device

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)
        full_pos = torch.cat([self.shared_pos[:, :self.grid_length, :],
                              self.shared_pos[:, :self.grid_length, :]], dim=1)

        # Add positional encoding
        full_seq = full_seq + full_pos

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # Transformer encode
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)

        return harmony_output
    # end forward

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SingleGridMLMelHarm

# Modular single encoder
class SEModular(nn.Module):
    def __init__(
            self, 
            chord_vocab_size,  # V
            d_model=512, 
            nhead=8, 
            num_layers=8, 
            dim_feedforward=2048,
            pianoroll_dim=13,      # PCP + bars only
            grid_length=80,
            condition_dim=None,  # if not None, add a condition token of this dim at start
            unmasking_stages=None,  # if not None, use stage-based unmasking
            trainable_pos_emb=False,
            dropout=0.3,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.grid_length = grid_length
        self.condition_dim = condition_dim
        self.unmasking_stages = unmasking_stages
        self.trainable_pos_emb = trainable_pos_emb

        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)

        # If using condition token, a linear projection
        if self.condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, d_model, device=self.device)
            self.seq_len = 1 + grid_length + grid_length
        else:
            self.seq_len = grid_length + grid_length
        
        # Positional embeddings
        if self.trainable_pos_emb:
            self.full_pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model, device=device))
            nn.init.trunc_normal_(self.full_pos, std=0.02)
        else:
            # # Positional embeddings (separate for clarity)
            self.shared_pos = sinusoidal_positional_encoding(
                grid_length + (self.condition_dim is not None), d_model, device
            )
            self.full_pos = torch.cat([self.shared_pos[:, :(self.grid_length + (self.condition_dim is not None)), :],
                              self.shared_pos[:, :self.grid_length, :]], dim=1)
        
        # If using unmasking stages, add an embedding layer
        if self.unmasking_stages is not None:
            assert isinstance(self.unmasking_stages, int) and self.unmasking_stages > 0, "unmasking_stages must be a positive integer"
            self.stage_embedding_dim = 64
            self.stage_embedding = nn.Embedding(self.unmasking_stages, self.stage_embedding_dim, device=self.device)
            # New projection layer to go from (d_model + stage_embedding_dim) → d_model
            self.stage_proj = nn.Linear(self.d_model + self.stage_embedding_dim, self.d_model, device=self.device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model=d_model, 
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

    def forward(self, melody_grid, harmony_tokens=None, conditioning_vec=None, stage_indices=None):
        """
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = melody_grid.size(0)
        device = self.device

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)
        if conditioning_vec is not None and self.condition_dim is not None:
            # Project condition: (B, d_model) → (B, 1, d_model)
            cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)
            full_seq = torch.cat([cond_emb, full_seq], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.full_pos

        if self.unmasking_stages is not None:
            # add stage embedding to harmony part
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

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        for layer in self.encoder.layers:
            self_attns.append(layer.last_attn_weights)
        return self_attns
    # end get_attention_maps
# end class SEModular