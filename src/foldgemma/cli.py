"""Typer CLI for FoldGemma."""

import importlib.metadata
import logging
import os
import sys
from collections.abc import Iterator

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from foldgemma.config import ModelSize, ModelType, OutputFormat

console = Console()


def _resolve_gpu_conflicts() -> None:
    """Resolve GPU memory allocation conflicts between TensorFlow and PyTorch."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
    except Exception:
        pass


# --------------------------------------

try:
    _metadata = importlib.metadata.metadata("gemma")
    _summary = _metadata["Summary"]
except Exception:
    _summary = "foldgemma CLI"

app = typer.Typer(
    name="gemma",
    help=_summary,
    rich_markup_mode="markdown",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:
    """Show version."""
    if value:
        try:
            version = importlib.metadata.version("gemma")
            console.print(f"[bold]foldgemma v{version}[/bold] - {_summary}")
        except Exception:
            console.print("foldgemma (unknown version)")
        raise typer.Exit()


def setup_logging(verbose: int) -> None:
    """Setup logging."""
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
    logging.getLogger("markdown_it").setLevel(logging.INFO)


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show the version and exit."
    ),
    verbose: int = typer.Option(0, "--verbose", "-V", count=True, help="Increase verbosity level"),
) -> None:
    """Global configuration for the CLI."""
    setup_logging(verbose)


@app.command()
def train(
    data_dir: str = typer.Argument(
        ..., help="Path to the directory containing binary shards (inputs.bin, targets.bin, etc.)"
    ),
    epochs: int = typer.Option(10, help="Number of epochs to train"),
    steps_per_epoch: int = typer.Option(1000, help="Steps per epoch"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Directory to save checkpoints"),
    model_type: ModelType = typer.Option(ModelType.GEMMA, help="Model type to train"),
    model_size: ModelSize = typer.Option(ModelSize.SMALL, help="Model size variant"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """🏋️ Train the FoldGemma model."""
    setup_logging(int(verbose))
    _resolve_gpu_conflicts()

    from foldgemma.config import FoldGemmaConfig
    from foldgemma.data.dataset import FoldGemmaDataset
    from foldgemma.trainer import FoldGemmaTrainer

    if not os.path.exists(data_dir):
        console.print(f"[red]:x: Error:[/red] Dataset path not found: {data_dir}")
        raise typer.Exit(1)

    dataset = FoldGemmaDataset(data_dir=data_dir)

    config_map = {
        ModelSize.SMALL: FoldGemmaConfig.small,
        ModelSize.BASE: FoldGemmaConfig.base,
        ModelSize.LARGE: FoldGemmaConfig.large,
    }
    config = config_map[model_size](model_type=model_type)

    trainer = FoldGemmaTrainer(config=config, learning_rate=learning_rate, model_type=model_type)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        console=console,
        transient=False,
    ) as progress:
        # We need a reference to the task so callbacks can update it
        task_id = progress.add_task(f"Training Epoch 1/{epochs}", total=steps_per_epoch, loss=0.0)

        def on_epoch_start(epoch: int, total_epochs: int) -> None:
            progress.update(task_id, description=f"Training Epoch {epoch + 1}/{total_epochs}", completed=0, loss=0.0)

        def on_step(step: int, loss: float) -> None:
            progress.update(task_id, advance=1, loss=loss)

        def on_epoch_end(epoch: int, avg_loss: float) -> None:
            console.print(f":white_check_mark: Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        trainer.fit(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            checkpoint_dir=checkpoint_dir,
            on_epoch_start=on_epoch_start,
            on_step=on_step,
            on_epoch_end=on_epoch_end,
        )

    console.print(":tada: Training complete!")


@app.command()
def infer(
    input: str = typer.Argument("-", help="Input FASTA file of amino acid sequences or `-` for stdin"),
    output: str = typer.Argument("-", help="Output FASTA file for 3di sequences or `-` for stdout"),
    model_type: ModelType = typer.Option(ModelType.GEMMA, help="Model type to infer"),
    model_size: ModelSize = typer.Option(ModelSize.SMALL, help="Model size variant"),
    weights: str | None = typer.Option(None, help="Path to safetensors weights"),
    output_format: OutputFormat = typer.Option(OutputFormat.FASTA, help="Output format for inference results"),
) -> None:
    """:brain: Run inference to predict 3di structures from AA sequences."""
    import torch
    from safetensors.torch import load_file

    from foldgemma import FoldGemma, FoldT5Gemma
    from foldgemma.config import FoldGemmaConfig
    from foldgemma.data.vocabulary import Protein3diVocabulary
    from foldgemma.io import read_fasta_bytes, write_fasta_bytes

    config_map = {
        ModelSize.SMALL: FoldGemmaConfig.small,
        ModelSize.BASE: FoldGemmaConfig.base,
        ModelSize.LARGE: FoldGemmaConfig.large,
    }
    config = config_map[model_size](model_type=model_type)

    if config.model_type == ModelType.T5GEMMA:
        model = FoldT5Gemma(config)
    else:
        model = FoldGemma(config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if weights:
        logging.getLogger("gemma").info(f":inbox_tray: Loading weights from {weights}...")
        state_dict = load_file(weights)
        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    vocab = Protein3diVocabulary()

    in_handle = sys.stdin.buffer if input == "-" else open(input, "rb")
    out_handle = sys.stdout.buffer if output == "-" else open(output, "wb")

    def process_sequences() -> Iterator[tuple[bytes, bytes]]:
        for header, seq_bytes in read_fasta_bytes(in_handle):
            input_ids = vocab.encode_bytes(seq_bytes)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type != "mps"
                else torch.autocast(device_type="cpu", enabled=False),
            ):
                if config.model_type == ModelType.T5GEMMA:
                    assert isinstance(model, FoldT5Gemma)
                    out_tensor = model.generate(input_tensor)
                else:
                    out_tensor = model(input_tensor)

            if config.model_type == ModelType.GEMMA:
                out_ids = out_tensor.argmax(dim=-1)[0].cpu().tolist()
            else:
                out_ids = out_tensor[0].cpu().tolist()

            out_bytes = vocab.decode_bytes(out_ids)
            out_bytes = out_bytes.replace(b"<pad>", b"").replace(b"<unk>", b"")
            yield header, out_bytes

    # Create an indefinite spinner for processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(":dna: Processing sequences...", total=None)

        def iter_with_progress() -> Iterator[tuple[bytes, bytes]]:
            for result in process_sequences():
                progress.advance(task_id, 1)
                yield result

        if output_format == OutputFormat.JSONL:
            import orjson

            for header, seq in iter_with_progress():
                record = {
                    "header": header.decode("utf-8", errors="replace"),
                    "sequence": seq.decode("utf-8", errors="replace"),
                }
                out_handle.write(orjson.dumps(record) + b"\n")
        else:
            write_fasta_bytes(out_handle, iter_with_progress())

    if input != "-":
        in_handle.close()
    if output != "-":
        out_handle.close()

    console.print(":white_check_mark: Inference complete.")


@app.command()
def prep(
    db_path: str = typer.Argument(..., help="Path prefix to Foldseek database (e.g. afdb50)"),
    out_dir: str = typer.Argument(..., help="Directory to output binary datasets"),
    num_workers: int = typer.Option(4, help="Number of parallel PyTorch DataLoader workers"),
    prefix: str | None = typer.Option(None, help="Prefix for the output dataset shards"),
) -> None:
    """:wrench: Prepare Steinegger Lab AFDB data into binary datasets for training."""
    from foldgemma.data.prep import write_dataset_from_foldseek

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.fields[records]} records processed"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(":wrench: Converting Foldseek to binary...", total=None, records=0)

        def on_progress(count: int) -> None:
            progress.update(task_id, advance=1, records=progress.tasks[task_id].fields["records"] + count)

        total = write_dataset_from_foldseek(
            db_prefix=db_path, out_dir=out_dir, num_workers=num_workers, prefix=prefix, progress_callback=on_progress
        )

        progress.update(task_id, completed=100)  # Indefinite to complete

    console.print(f":white_check_mark: Data prep complete! Successfully serialized {total} records to binary datasets.")


@app.command()
def deploy(
    repo_id: str = typer.Argument(..., help="Target Hugging Face repository ID (e.g. username/foldgemma)"),
    model_path: str = typer.Argument("./model.safetensors", help="Path to the model file"),
    token: str | None = typer.Option(None, help="HF API token. Falls back to HF_TOKEN env var if not set."),
) -> None:
    """:rocket: Deploy a trained model to the Hugging Face Hub."""
    from foldgemma.deploy import deploy_to_huggingface

    try:
        # The deploy script already logs nicely, but we can wrap it in a spinner
        with console.status(f"[bold green]Deploying {model_path} to {repo_id}..."):
            deploy_to_huggingface(repo_id=repo_id, model_path=model_path, token=token)
        console.print(":white_check_mark: Deployment complete!")
    except Exception as e:
        console.print(f"[bold red]:x: Deployment failed:[/bold red] {e}")
        raise typer.Exit(1)


def _entrypoint() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    _entrypoint()
