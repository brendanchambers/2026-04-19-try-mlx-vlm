#!/usr/bin/env python3
"""
MLX-VLM Demo: Image Understanding with Vision-Language Models

This script demonstrates how mlx-vlm analyzes images and generates text descriptions.
Note: mlx-vlm is for image-to-text (understanding images), NOT text-to-image generation.
"""

import json
from datetime import datetime
from pathlib import Path
import urllib.request

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table


# Configuration
MODEL_NAME = "google/gemma-4-e2b-it"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = "Describe this image in detail, focusing on the main subjects and their activities."
MAX_TOKENS = 500
TEMPERATURE = 0.7


def setup_request_directory(console: Console) -> tuple[str, Path]:
    """Create timestamped request directory and display header."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    request_dir = Path("requests") / timestamp
    request_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"[bold cyan]MLX-VLM Image Analysis[/bold cyan]\n"
        f"Model: [yellow]{MODEL_NAME}[/yellow]\n"
        f"Request ID: [green]{timestamp}[/green]",
        border_style="cyan"
    ))

    return timestamp, request_dir


def download_image(image_url: str, request_dir: Path, console: Console) -> Path:
    """Download image from URL and save to request directory."""
    image_path = request_dir / "image.jpg"
    console.print(f"\n[cyan]Downloading image...[/cyan]")
    urllib.request.urlretrieve(image_url, image_path)
    console.print(f"[green]✓[/green] Image saved to: {image_path}")
    return image_path


def load_vlm_model(console: Console):
    """Load the vision-language model and processor."""
    console.print(f"\n[cyan]Loading model...[/cyan]")
    model, processor = load(MODEL_NAME)
    config = load_config(MODEL_NAME)
    console.print(f"[green]✓[/green] Model loaded")
    return model, processor, config


def analyze_image(model, processor, config, image_url: str, prompt: str, console: Console):
    """Run model inference on the image."""
    console.print(f"\n[cyan]Analyzing image...[/cyan]")
    console.print(f"Prompt: [italic]{prompt}[/italic]\n")

    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1
    )

    output = generate(
        model,
        processor,
        formatted_prompt,
        [image_url],
        verbose=False,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return output


def create_output_data(output, timestamp: str) -> dict:
    """Extract and structure model output into a dictionary."""
    return {
        "text": output.text,
        "token": int(output.token),
        "prompt_tokens": int(output.prompt_tokens),
        "generation_tokens": int(output.generation_tokens),
        "total_tokens": int(output.total_tokens),
        "prompt_tps": float(output.prompt_tps),
        "generation_tps": float(output.generation_tps),
        "peak_memory": float(output.peak_memory),
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "image_url": IMAGE_URL,
        "prompt": PROMPT,
    }


def save_output(output_data: dict, request_dir: Path, console: Console) -> Path:
    """Save output data to JSON file."""
    output_file = request_dir / "output.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"[green]✓[/green] Output saved to: {output_file}\n")
    return output_file


def display_results(output_data: dict, request_dir: Path, console: Console):
    """Display formatted results using rich."""
    # Display model response
    console.print(Panel(
        output_data["text"],
        title="[bold yellow]Model Response[/bold yellow]",
        border_style="yellow",
        padding=(1, 2)
    ))

    # Display performance metrics
    table = Table(title="Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Prompt Tokens", str(output_data["prompt_tokens"]))
    table.add_row("Generation Tokens", str(output_data["generation_tokens"]))
    table.add_row("Total Tokens", str(output_data["total_tokens"]))
    table.add_row("Prompt Speed", f"{output_data['prompt_tps']:.1f} tokens/sec")
    table.add_row("Generation Speed", f"{output_data['generation_tps']:.1f} tokens/sec")
    table.add_row("Peak Memory", f"{output_data['peak_memory']:.2f} GB")

    console.print("\n")
    console.print(table)

    # Display JSON output
    console.print("\n[bold cyan]Full JSON Output:[/bold cyan]")
    console.print(JSON(json.dumps(output_data, indent=2)))

    console.print(f"\n[green]✓ Analysis complete![/green]")
    console.print(f"[dim]Results saved to: {request_dir}[/dim]")


def main():
    """Analyze an image using mlx-vlm and generate a text description."""
    console = Console()

    timestamp, request_dir = setup_request_directory(console)
    download_image(IMAGE_URL, request_dir, console)
    model, processor, config = load_vlm_model(console)
    output = analyze_image(model, processor, config, IMAGE_URL, PROMPT, console)
    output_data = create_output_data(output, timestamp)
    save_output(output_data, request_dir, console)
    display_results(output_data, request_dir, console)


if __name__ == "__main__":
    main()
