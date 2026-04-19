#!/usr/bin/env python3
"""
MLX-VLM Demo: Image Understanding with Vision-Language Models

This script demonstrates how mlx-vlm analyzes images and generates text descriptions.
Note: mlx-vlm is for image-to-text (understanding images), NOT text-to-image generation.
"""

# Configuration
MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = "Describe this image in detail, focusing on the main subjects and their activities."
MAX_TOKENS = 200
TEMPERATURE = 0.7


def main():
    """Analyze an image using mlx-vlm and generate a text description."""
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"Loading model: {MODEL_NAME}")
    print("This may take a few minutes on first run as it downloads the model...")

    # Load model and processor
    config = load_config(MODEL_NAME)
    model, processor = load(MODEL_NAME, {"trust_remote_code": True})

    print(f"\nAnalyzing image: {IMAGE_URL}")
    print(f"Prompt: {PROMPT}\n")

    # Prepare the prompt with the image
    formatted_prompt = apply_chat_template(
        processor, config, PROMPT, num_images=1
    )

    # Generate description
    output = generate(
        model,
        processor,
        IMAGE_URL,
        formatted_prompt,
        verbose=False,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    print("=" * 80)
    print("Model Response:")
    print("=" * 80)
    print(output)
    print("=" * 80)

    print("\n✓ Demo complete!")
    print("\nWhat mlx-vlm does:")
    print("  • Analyzes images and generates text descriptions")
    print("  • Answers questions about images")
    print("  • Supports multi-modal understanding (text, images, audio, video)")
    print("\nWhat mlx-vlm does NOT do:")
    print("  • Generate images from text (that requires stable-diffusion-mlx)")


if __name__ == "__main__":
    main()
