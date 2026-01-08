#!/usr/bin/env python3
"""Export MobileCLIP-S1 visual encoder to ONNX and pre-compute nursing text embeddings."""

import os
import struct
import numpy as np
import torch
import open_clip
from pathlib import Path

NURSING_LABELS = {
    "position": [
        "a patient lying flat on their back in a hospital bed",
        "a patient lying on their left side in a hospital bed",
        "a patient lying on their right side in a hospital bed",
        "a patient lying face down on their stomach in a hospital bed",
        "a patient sitting up in a hospital bed",
        "a patient sitting in a chair or wheelchair",
        "a patient standing upright",
        "a person lying on the floor, possibly fallen",
    ],
    "alertness": [
        "a patient who is awake and alert, eyes open, looking around",
        "a patient who appears drowsy or sleepy, eyes half-closed",
        "a patient who is sleeping peacefully with eyes closed",
        "a patient with eyes closed, resting",
        "an unresponsive patient, not reacting to surroundings",
    ],
    "activity": [
        "a patient lying completely still with no movement",
        "a patient with slight movement, small gestures",
        "a patient moving moderately, shifting position",
        "a patient moving actively, gesturing or repositioning",
        "a patient who appears restless or agitated",
    ],
    "comfort": [
        "a patient who appears comfortable and relaxed",
        "a patient showing signs of mild discomfort",
        "a patient showing moderate discomfort or pain",
        "a patient who appears distressed or in significant pain",
        "a patient with facial expressions indicating pain",
    ],
    "safety": [
        "a patient in a safe, normal hospital room setting",
        "a patient at risk of falling, near edge of bed",
        "a patient who has fallen on the floor",
        "a patient attempting to get out of bed",
        "medical equipment that appears disconnected or problematic",
    ],
}


def main():
    output_dir = Path("app/src/main/assets/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MobileCLIP-S1 Export for Triage Vision Android")
    print("=" * 60)

    print("Loading MobileCLIP-S1 model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP-S1',
        pretrained='datacompdr'
    )
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
    model.eval()
    
    img_size = model.visual.image_size[0]
    params = sum(p.numel() for p in model.visual.parameters()) / 1e6
    print(f"Model loaded - image size: {img_size}x{img_size}, visual params: {params:.1f}M")

    # Export visual encoder to ONNX
    temp_path = output_dir / "temp_visual.onnx"
    print(f"\nExporting visual encoder...")
    
    dummy_input = torch.randn(1, 3, img_size, img_size)
    visual_encoder = model.visual
    
    torch.onnx.export(
        visual_encoder,
        dummy_input,
        str(temp_path),
        input_names=['input'],
        output_names=['embedding'],
        opset_version=18,
        do_constant_folding=True,
    )
    
    # Load and save as single file
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    
    onnx_model = onnx.load(str(temp_path))
    load_external_data_for_model(onnx_model, str(temp_path.parent))
    
    final_path = output_dir / "mobileclip_visual.onnx"
    onnx.save(onnx_model, str(final_path))
    
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"  Visual encoder: {size_mb:.1f} MB")
    
    # Clean up temp files
    os.remove(temp_path)
    for f in output_dir.glob("*.data"):
        os.remove(f)

    # Compute text embeddings
    print("\nComputing text embeddings...")
    embeddings = {}
    with torch.no_grad():
        for category, prompts in NURSING_LABELS.items():
            tokens = tokenizer(prompts)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings[category] = text_features.cpu().numpy()
            print(f"  {category}: {embeddings[category].shape}")

    # Save embeddings as binary
    embeddings_path = output_dir / "nursing_text_embeddings.bin"
    print(f"\nSaving embeddings to: {embeddings_path}")
    
    with open(embeddings_path, 'wb') as f:
        f.write(struct.pack('I', len(embeddings)))
        for category, emb_array in embeddings.items():
            cat_bytes = category.encode('utf-8')
            f.write(struct.pack('I', len(cat_bytes)))
            f.write(cat_bytes)
            num_labels, embedding_dim = emb_array.shape
            f.write(struct.pack('II', num_labels, embedding_dim))
            f.write(emb_array.astype(np.float32).tobytes())

    size_kb = os.path.getsize(embeddings_path) / 1024
    print(f"  Embeddings saved: {size_kb:.1f} KB")

    # Get embedding dim for config
    sample_emb = list(embeddings.values())[0]
    emb_dim = sample_emb.shape[1]

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Visual encoder: {final_path} ({size_mb:.1f} MB)")
    print(f"  Text embeddings: {embeddings_path} ({size_kb:.1f} KB)")
    print(f"  Config: input={img_size}x{img_size}, embed_dim={emb_dim}")
    print("=" * 60)


if __name__ == "__main__":
    main()
