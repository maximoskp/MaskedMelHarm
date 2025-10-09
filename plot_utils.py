import matplotlib.pyplot as plt
import os
import numpy as np

def save_attention_maps_with_split(encoder, melody_len, save_dir="attn_maps", prefix="layer"):
    """
    Save attention maps for all layers and all heads, plus an overlay per layer.

    Args:
        encoder: Transformer encoder with custom layers storing `last_attn_weights`
        melody_len: int, number of melody tokens before harmony tokens start
        save_dir: directory to save images
        prefix: prefix for filenames
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, layer in enumerate(encoder.layers):
        attn = layer.last_attn_weights  # shape [B, H, L, L]
        if attn is None:
            print(f"⚠️ No attention stored in layer {i}. Did you run a forward pass?")
            continue

        B, H, L, _ = attn.shape

        # ---- Per-head plots ----
        for h in range(H):
            attn_map = attn[0, h].cpu().numpy()  # take first batch

            plt.figure(figsize=(6, 5))
            plt.imshow(attn_map, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"Attention Map - Layer {i}, Head {h}")
            plt.xlabel("Key/Value positions")
            plt.ylabel("Query positions")

            # boundary lines
            plt.axvline(x=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)
            plt.axhline(y=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)

            plt.tight_layout()
            fname = os.path.join(save_dir, f"{prefix}_L{i}_H{h}.png")
            plt.savefig(fname)
            plt.close()

        # ---- Overlay / aggregate plot ----
        # Option 1: average across heads
        attn_avg = attn[0].mean(dim=0).cpu().numpy()

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_avg, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(f"Attention Map - Layer {i} (Avg across {H} heads)")
        plt.xlabel("Key/Value positions")
        plt.ylabel("Query positions")

        # boundary lines
        plt.axvline(x=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)
        plt.axhline(y=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)

        plt.tight_layout()
        fname = os.path.join(save_dir, f"{prefix}_L{i}_ALL.png")
        plt.savefig(fname)
        plt.close()

# end save_attention_maps_with_split