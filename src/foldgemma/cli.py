import os
import sys
import glob
from typing import List, Optional
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import logging

def _resolve_gpu_conflicts():
    """Resolve GPU memory allocation conflicts between TensorFlow and PyTorch."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
    except Exception:
        pass

    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
# --------------------------------------

app = typer.Typer(help="FoldGemma: Protein folding language models", rich_markup_mode="markdown")
console = Console()

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
    )

@app.command()
def train(
    tfrecord: List[str] = typer.Argument(..., help="Path(s) or glob pattern to the training TFRecord file(s)"),
    epochs: int = typer.Option(10, help="Number of epochs to train"),
    steps_per_epoch: int = typer.Option(1000, help="Steps per epoch"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Directory to save checkpoints"),
    model_type: str = typer.Option("foldgemma", help="Model type to train"),
    model_size: str = typer.Option("small", help="Model size variant"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """📈 Train the FoldGemma model."""
    setup_logging(verbose)
    _resolve_gpu_conflicts()
    
    from foldgemma.trainer import FoldGemmaTrainer
    from foldgemma.config import FoldGemmaConfig, ModelType
    from foldgemma.data.pipeline import FoldGemmaDataPipeline

    # Resolve globs
    resolved_tfrecords = []
    for path_arg in tfrecord:
        if "*" in path_arg or "?" in path_arg:
            resolved_tfrecords.extend(glob.glob(path_arg))
        else:
            resolved_tfrecords.append(path_arg)
    resolved_tfrecords = sorted(list(set(resolved_tfrecords)))

    if not resolved_tfrecords:
        console.print(f"[red]❌ Error:[/red] No TFRecord files found matching {tfrecord}")
        raise typer.Exit(1)

    pipeline = FoldGemmaDataPipeline(
        tfrecord_path=resolved_tfrecords,
        batch_size=batch_size,
    )

    if model_size == "small":
        config = FoldGemmaConfig.small(model_type=ModelType(model_type))
    elif model_size == "base":
        config = FoldGemmaConfig.base(model_type=ModelType(model_type))
    else:
        config = FoldGemmaConfig.large(model_type=ModelType(model_type))

    trainer = FoldGemmaTrainer(
        config=config,
        learning_rate=learning_rate,
        model_type=ModelType(model_type)
    )

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

        def on_epoch_start(epoch, total_epochs):
            progress.update(task_id, description=f"Training Epoch {epoch+1}/{total_epochs}", completed=0, loss=0.0)

        def on_step(step, loss):
            progress.update(task_id, advance=1, loss=loss)

        def on_epoch_end(epoch, avg_loss):
            console.print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        trainer.fit(
            pipeline=pipeline,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            checkpoint_dir=checkpoint_dir,
            on_epoch_start=on_epoch_start,
            on_step=on_step,
            on_epoch_end=on_epoch_end
        )
    
    console.print("🎉 Training complete!")

@app.command()
def infer(
    input: str = typer.Option("-", "--input", "-i", help="Input FASTA file of amino acid sequences"),
    output: str = typer.Option("-", "--output", "-o", help="Output FASTA file for 3di sequences"),
    model_type: str = typer.Option("foldgemma", help="Model type to infer"),
    model_size: str = typer.Option("small", help="Model size variant"),
    weights: Optional[str] = typer.Option(None, help="Path to safetensors weights"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """🧠 Run inference to predict 3di structures from AA sequences."""
    setup_logging(verbose)
    
    from foldgemma import FoldGemma, FoldGemmaT5
    from foldgemma.config import FoldGemmaConfig, ModelType
    from foldgemma.data.vocabulary import Protein3diVocabulary
    from foldgemma.io import read_fasta_bytes, write_fasta_bytes
    import torch
    from safetensors.torch import load_file

    if model_size == "small":
        config = FoldGemmaConfig.small(model_type=ModelType(model_type))
    elif model_size == "base":
        config = FoldGemmaConfig.base(model_type=ModelType(model_type))
    else:
        config = FoldGemmaConfig.large(model_type=ModelType(model_type))

    if config.model_type == ModelType.FOLDGEMMA_T5:
        model = FoldGemmaT5(config)
    else:
        model = FoldGemma(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if weights:
        logging.getLogger("foldgemma").info(f"📥 Loading weights from {weights}...")
        state_dict = load_file(weights)
        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    vocab = Protein3diVocabulary()
    
    in_handle = sys.stdin.buffer if input == "-" else open(input, "rb")
    out_handle = sys.stdout.buffer if output == "-" else open(output, "wb")

    def process_sequences():
        for header, seq_bytes in read_fasta_bytes(in_handle):
            input_ids = vocab.encode_bytes(seq_bytes)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type != "mps" else torch.autocast(device_type="cpu", enabled=False):
                if config.model_type == ModelType.FOLDGEMMA_T5:
                    out_tensor = model.generate(input_tensor)
                else:
                    out_tensor = model(input_tensor)

            if config.model_type == ModelType.FOLDGEMMA:
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
        task_id = progress.add_task("🧬 Processing sequences...", total=None)
        
        def iter_with_progress():
            for result in process_sequences():
                progress.advance(task_id, 1)
                yield result
                
        write_fasta_bytes(out_handle, iter_with_progress())
        
    if input != "-":
        in_handle.close()
    if output != "-":
        out_handle.close()

    console.print("✅ Inference complete.")

@app.command()
def prep(
    db_path: str = typer.Argument(..., help="Path prefix to Foldseek database (e.g. afdb50)"),
    out_dir: str = typer.Argument(..., help="Directory to output TFRecords"),
    num_workers: int = typer.Option(4, help="Number of parallel PyTorch DataLoader workers"),
    prefix: Optional[str] = typer.Option(None, help="Prefix for the output TFRecord files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """🔧 Prepare Steinegger Lab AFDB data into TFRecords for training."""
    setup_logging(verbose)
    from foldgemma.data.prep import write_tfrecords_from_foldseek

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.fields[records]} records processed"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("🔧 Converting Foldseek to TFRecords...", total=None, records=0)
        
        def on_progress(count):
            progress.update(task_id, advance=1, records=progress.tasks[task_id].fields['records'] + count)

        total = write_tfrecords_from_foldseek(
            db_prefix=db_path, 
            out_dir=out_dir, 
            num_workers=num_workers, 
            prefix=prefix,
            progress_callback=on_progress
        )
        
        progress.update(task_id, completed=100) # Indefinite to complete

    console.print(f"✅ Data prep complete! Successfully serialized {total} records to TFRecords.")

@app.command()
def deploy(
    repo_id: str = typer.Option(..., help="Target Hugging Face repository ID (e.g. username/foldgemma)"),
    model_path: str = typer.Option("./model.safetensors", help="Path to the model file"),
    token: Optional[str] = typer.Option(None, help="HF API token. Falls back to HF_TOKEN env var if not set."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """🚀 Deploy a trained model to the Hugging Face Hub."""
    setup_logging(verbose)
    from foldgemma.deploy import deploy_to_huggingface

    try:
        # The deploy script already logs nicely, but we can wrap it in a spinner
        with console.status(f"[bold green]Deploying {model_path} to {repo_id}..."):
            deploy_to_huggingface(repo_id=repo_id, model_path=model_path, token=token)
        console.print("✅ Deployment complete!")
    except Exception as e:
        console.print(f"[bold red]❌ Deployment failed:[/bold red] {e}")
        raise typer.Exit(1)

def main():
    app()

if __name__ == "__main__":
    main()
