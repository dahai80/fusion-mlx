#!/usr/bin/env python3
"""Quick test of the fusion_gui API endpoints."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fusion_gui.huggingface_integration import get_huggingface_client


async def test_hf_integration():
    """Test HuggingFace integration directly."""
    print("🔍 Testing HuggingFace Integration...")

    hf_client = get_huggingface_client()

    # Test popular models
    print("\n📈 Getting popular models...")
    models = hf_client.get_popular_mlx_models(limit=3)
    for model in models:
        print(f"  - {model.id} ({model.model_type}, {model.downloads:,} downloads)")

    # Test model categories
    print("\n📂 Getting model categories...")
    categories = hf_client.get_model_categories()
    for category, model_list in categories.items():
        if model_list:
            print(f"  {category}: {len(model_list)} models")
            if model_list:
                print(f"    Example: {model_list[0]}")

    # Test model details
    if models:
        model_id = models[0].id
        print(f"\n🔍 Getting details for {model_id}...")
        details = hf_client.get_model_details(model_id)
        if details:
            print(f"  Name: {details.name}")
            print(f"  Author: {details.author}")
            print(f"  Type: {details.model_type}")
            print(
                f"  Size: {details.size_gb}GB" if details.size_gb else "  Size: Unknown"
            )
            print(
                f"  Memory: {details.estimated_memory_gb}GB"
                if details.estimated_memory_gb
                else "  Memory: Unknown"
            )
            print(f"  MLX Compatible: {details.mlx_compatible}")

    print("\n✅ HuggingFace integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_hf_integration())
