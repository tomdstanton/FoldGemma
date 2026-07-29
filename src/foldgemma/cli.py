import argparse
import os
import re
import sys
from abc import ABC
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import IO



# --- Global GPU Conflict Resolution ---
# PyTorch and TensorFlow share the same CUDA runtime library (cudart) in the process.
# We cannot use CUDA_VISIBLE_DEVICES="-1" because if TF initializes it first, PyTorch will never see GPUs.
# If PyTorch initializes it first, TF will see all GPUs and might crash when attempting to share them.
# The ONLY safe way is to tell TensorFlow's logical device manager to ignore all physical GPUs.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import torch
    if torch.cuda.is_available():
        # Force PyTorch to claim the CUDA runtime symbols before TensorFlow is imported
        torch.cuda.init()
except Exception:
    pass

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
# --------------------------------------

# Classes --------------------------------------------------------------------------------------------------------------
class Colors:
    """A non-instantiable namespace for ANSI escape sequences."""

    ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def __init__(self):
        raise TypeError("The Colors class is a namespace and cannot be instantiated.")

    RESET = "\033[0m"
    BOLD = "\033[1m"
    BOLD_RED = "\033[1;31m"
    BOLD_CYAN = "\033[1;36m"

    @classmethod
    def wrap(cls, text: str | None, *styles: str) -> str:
        """Wraps text in the specified color(s)/style(s) and appends the reset sequence."""
        if not text:
            return ""
        if not cls.ENABLED:
            return text
        return f"{''.join(styles)}{text}{cls.RESET}"


class FoldGemmaHelpFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter to add ANSI colors to argparse output."""

    def _format_usage(self, usage, actions, groups, prefix):
        positionals = [a for a in actions if not a.option_strings]
        result = super()._format_usage(usage, positionals, groups, prefix)
        result = re.sub(r"\{[a-zA-Z0-9_,\.-]+\}", Colors.wrap("[subcommand]", Colors.BOLD_CYAN), result)
        
        actual_prefix = prefix if prefix is not None else "usage: "
        target = f"{actual_prefix}{self._prog}"
        
        if result.startswith(target):
            if any(a.option_strings for a in actions):
                colored_options = Colors.wrap("[options]", Colors.BOLD_CYAN)
                result = result.replace(target, f"{target} {colored_options}", 1)
            result = result.replace(actual_prefix, Colors.wrap(actual_prefix, Colors.BOLD_CYAN), 1)
            
        return result

    def start_section(self, heading):
        if heading:
            heading = Colors.wrap(heading, Colors.BOLD_CYAN)
        super().start_section(heading)

    def _format_action(self, action):
        result = super()._format_action(action)
        if type(action).__name__ == "_SubParsersAction":
            lines = result.split("\n", 1)
            if len(lines) > 1:
                result = lines[1]
        return result


class HelpOnErrorParser(argparse.ArgumentParser):
    """An ArgumentParser that prints full help on error."""
    
    def error(self, message):
        if match := re.search(r"invalid choice: '?([^']+)'? \(choose from (.*)\)", message):
            invalid = match.group(1)
            choices = [c.strip("'").strip() for c in match.group(2).split(", ")]
            from difflib import get_close_matches
            if matches := get_close_matches(invalid, choices):
                message += f"\n    💡 Did you mean '{Colors.wrap(matches[0], Colors.BOLD_CYAN)}'?"
                
        self.print_help(sys.stderr)
        self.exit(2, f"\n{Colors.wrap('❌ Error:', Colors.BOLD_RED)} {message}\n")


class Cli:
    """Class defining the root FoldGemma CLI."""

    def __init__(self, description: str | None = None, epilog: str | None = None):
        self.verbose = False
        self.global_parser = HelpOnErrorParser(add_help=False)
        self.global_parser.add_argument(
            "-V", "--verbose", action="store_true", help="Enable verbose output/progress"
        )

        self.parser = HelpOnErrorParser(
            description=Colors.wrap(description, Colors.BOLD) if description else description,
            epilog=Colors.wrap(epilog, Colors.BOLD) if epilog else epilog,
            parents=[self.global_parser],
            formatter_class=FoldGemmaHelpFormatter,
        )

        try:
            version = metadata.version("foldgemma")
        except metadata.PackageNotFoundError:
            version = "unknown"

        self.parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"%(prog)s {version}",
            help="Show program's version number and exit",
        )

        if hasattr(self.parser, "_optionals"):
            self.parser._optionals.title = Colors.wrap("🌎 Global options", Colors.BOLD)

        self.subparsers = self.parser.add_subparsers(
            title=Colors.wrap("💬 Commands", Colors.BOLD), dest="command", required=True
        )
        self._open_handles = []

    def add_command(self, command: "Command"):
        command.cli = self
        command.build(self.subparsers, parent_parsers=[self.global_parser])

    def run(self, args: list[str] | None = None):
        parsed_args = self.parser.parse_args(args)
        self.verbose = getattr(parsed_args, "verbose", False)

        if hasattr(parsed_args, "func"):
            parsed_args.func(parsed_args)
        else:
            self.parser.print_help()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if exc_type is KeyboardInterrupt:
            self.msg("\n🛑 Cancelled by user.")
            sys.exit(1)
        elif exc_type is BrokenPipeError:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(130)
        elif exc_type is PermissionError:
            self.msg(f"🔒 Permission denied: {exc_val}")
            sys.exit(1)
        elif exc_type is FileNotFoundError:
            self.msg(f"📄 File not found: {exc_val}")
            sys.exit(1)

    def exit(self, msg: str, code: int = 1):
        self.msg(f"❌ {msg}")
        sys.exit(code)

    def cleanup(self):
        for handle in self._open_handles:
            if handle not in (sys.stdout, sys.stdin, sys.stderr, sys.stdout.buffer, sys.stdin.buffer, sys.stderr.buffer):
                handle.close()
        self._open_handles.clear()

    def msg(self, msg: str | None, **kwargs) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, **kwargs)

    def progress(self, iterable: Iterable, msg: str) -> Iterable:
        try:
            total = len(iterable)
        except TypeError:
            total = "?"

        for i, item in enumerate(iterable, start=1):
            if self.verbose:
                print(f"\r{msg} {i}/{total}", end="", file=sys.stderr, flush=True)
            yield item

        if self.verbose:
            print(file=sys.stderr)

    def open_file(self, file: str | Path, mode: str = "rb") -> IO:
        file = str(file)
        if file == "-" or file == "stdout":
            return sys.stdout.buffer if "b" in mode else sys.stdout
        if file == "stdin":
            return sys.stdin.buffer if "b" in mode else sys.stdin

        handle = open(file, mode)
        self._open_handles.append(handle)
        return handle


class Command(ABC):
    name: str = ""
    aliases: list[str] = []
    description: str = ""
    help_text: str = ""

    def __init__(self):
        self.parser: argparse.ArgumentParser | None = None
        self.subcommands: list["Command"] = []
        self.cli: Cli | None = None

        if not self.name:
            self.name = type(self).__name__.lower()
        if not self.description:
            if type(self).__doc__ and type(self).__doc__ != Command.__doc__:
                self.description = type(self).__doc__
        if not self.help_text and self.description:
            self.help_text = self.description.strip().split("\n")[0]

        self.register_subcommands()

    def register_subcommands(self):
        pass

    def setup_arguments(self):
        pass

    def get_shared_parser(self) -> argparse.ArgumentParser | None:
        return None

    def __call__(self, args: argparse.Namespace):
        pass

    def build(self, subparsers: argparse._SubParsersAction, parent_parsers: list[argparse.ArgumentParser] | None = None):
        parents = parent_parsers or []
        self.parser = subparsers.add_parser(
            name=self.name,
            aliases=self.aliases,
            description=Colors.wrap(self.description, Colors.BOLD),
            help=self.help_text or self.description,
            parents=parents,
            formatter_class=FoldGemmaHelpFormatter,
        )

        self.setup_arguments()

        if hasattr(self.parser, "_optionals"):
            self.parser._optionals.title = Colors.wrap("🌎 Global options", Colors.BOLD)
            groups = self.parser._action_groups
            if self.parser._optionals in groups:
                groups.append(groups.pop(groups.index(self.parser._optionals)))

        if type(self).__call__ != Command.__call__:
            self.parser.set_defaults(func=self.__call__)

        if self.subcommands:
            is_required = type(self).__call__ == Command.__call__
            sub_action = self.parser.add_subparsers(
                title=Colors.wrap(f"'{self.name}' subcommands", Colors.BOLD),
                dest=f"{self.name}_subcommand",
                required=is_required,
            )

            child_parents = parents.copy()
            if shared := self.get_shared_parser():
                child_parents.append(shared)

            for cmd in self.subcommands:
                cmd.cli = self.cli
                cmd.build(sub_action, parent_parsers=child_parents)


# Commands -------------------------------------------------------------------------------------------------------------
class Train(Command):
    """🏋️ Train the FoldGemma model."""

    def setup_arguments(self):
        opts = self.parser.add_argument_group("Inputs")
        opts.add_argument(
            "tfrecord", 
            type=str, 
            nargs="+",
            help="Path(s) or glob pattern to the training TFRecord file(s)"
        )
        opts = self.parser.add_argument_group("Hyperparameters")
        opts.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
        opts.add_argument("--steps-per-epoch", type=int, default=1000, help="Steps per epoch (default: 1000)")
        opts.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
        opts.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
        opts.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
        opts.add_argument("--model-type", type=str, default="foldgemma", choices=["foldgemma", "foldgemma_t5"], help="Model type to train")
        opts.add_argument("--model-size", type=str, default="small", choices=["small", "base", "large"], help="Model size variant")

    def __call__(self, args: argparse.Namespace):
        print("DEBUG: Inside Train.__call__, beginning imports...", flush=True)
        from foldgemma.trainer import FoldGemmaTrainer
        from foldgemma.config import FoldGemmaConfig, ModelType
        from foldgemma.data.pipeline import FoldGemmaDataPipeline
        print("DEBUG: Imports complete.", flush=True)
        import glob
        
        tfrecords = []
        for path_arg in args.tfrecord:
            if "*" in path_arg or "?" in path_arg:
                tfrecords.extend(glob.glob(path_arg))
            else:
                tfrecords.append(path_arg)
        tfrecords = sorted(list(set(tfrecords)))
        
        if not tfrecords:
            self.cli.exit(f"No TFRecord files found matching {args.tfrecord}")
        
        print("DEBUG: Initializing DataPipeline...", flush=True)
        pipeline = FoldGemmaDataPipeline(
            tfrecord_path=tfrecords,
            batch_size=args.batch_size,
        )
        
        print("DEBUG: Creating Config...", flush=True)
        if args.model_size == "small":
            config = FoldGemmaConfig.small(model_type=ModelType(args.model_type))
        elif args.model_size == "base":
            config = FoldGemmaConfig.base(model_type=ModelType(args.model_type))
        else:
            config = FoldGemmaConfig.large(model_type=ModelType(args.model_type))

        print("DEBUG: Instantiating FoldGemmaTrainer...", flush=True)
        trainer = FoldGemmaTrainer(
            config=config,
            learning_rate=args.learning_rate,
            model_type=ModelType(args.model_type)
        )
        
        print("DEBUG: Calling trainer.fit()...", flush=True)
        trainer.fit(
            pipeline=pipeline,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            checkpoint_dir=args.checkpoint_dir
        )
        self.cli.msg(f"✅ Training complete.")


class Infer(Command):
    """🧠 Run inference to predict 3di structures from AA sequences."""

    def setup_arguments(self):
        opts = self.parser.add_argument_group("Inputs/Outputs")
        opts.add_argument(
            "-i", "--input", 
            default="-", 
            help="Input FASTA file of amino acid sequences (default: stdin)"
        )
        opts.add_argument(
            "-o", "--output", 
            default="-", 
            help="Output FASTA file for 3di sequences (default: stdout)"
        )
        opts.add_argument("--model-type", type=str, default="foldgemma", choices=["foldgemma", "foldgemma_t5"], help="Model type to infer")
        opts.add_argument("--model-size", type=str, default="small", choices=["small", "base", "large"], help="Model size variant")
        opts.add_argument("--weights", type=str, default=None, help="Path to safetensors weights")

    def __call__(self, args: argparse.Namespace):
        self.cli.msg(f"🧠 Initializing FoldGemma inference engine...")
        
        from foldgemma import FoldGemma, FoldGemmaT5
        from foldgemma.config import FoldGemmaConfig, ModelType
        from foldgemma.data.vocabulary import Protein3diVocabulary
        from foldgemma.io import read_fasta_bytes, write_fasta_bytes
        import torch
        from safetensors.torch import load_file

        if args.model_size == "small":
            config = FoldGemmaConfig.small(model_type=ModelType(args.model_type))
        elif args.model_size == "base":
            config = FoldGemmaConfig.base(model_type=ModelType(args.model_type))
        else:
            config = FoldGemmaConfig.large(model_type=ModelType(args.model_type))
        
        # Core Library API: Just instantiate the PyTorch modules directly!
        if config.model_type == ModelType.FOLDGEMMA_T5:
            model = FoldGemmaT5(config)
        else:
            model = FoldGemma(config)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        if args.weights:
            self.cli.msg(f"📥 Loading weights from {args.weights}...")
            state_dict = load_file(args.weights)
            # Cast dict to match model dtype
            state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
            
        vocab = Protein3diVocabulary()

        in_handle = self.cli.open_file(args.input, "rb")
        out_handle = self.cli.open_file(args.output, "wb")
        
        def process_sequences():
            for header, seq_bytes in self.cli.progress(read_fasta_bytes(in_handle), "🧬 Processing sequences..."):
                input_ids = vocab.encode_bytes(seq_bytes)
                input_tensor = torch.tensor([input_ids], dtype=torch.long)
                
                input_tensor = input_tensor.to(device)
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type != "mps" else torch.autocast(device_type="cpu", enabled=False):
                    if config.model_type == ModelType.FOLDGEMMA_T5:
                        out_tensor = model.generate(input_tensor)
                    else:
                        out_tensor = model(input_tensor)
                    
                # Assuming out_tensor shape [1, seq_len, vocab_size] for FoldGemma, we need argmax
                if config.model_type == ModelType.FOLDGEMMA:
                    out_ids = out_tensor.argmax(dim=-1)[0].cpu().tolist()
                else:
                    out_ids = out_tensor[0].cpu().tolist() # generate might already return ids
                    
                out_bytes = vocab.decode_bytes(out_ids)
                # Remove padding/unk if necessary. Actually we want everything up to pad
                out_bytes = out_bytes.replace(b"<pad>", b"").replace(b"<unk>", b"")
                
                yield header, out_bytes
                
        write_fasta_bytes(out_handle, process_sequences())
        self.cli.msg("✅ Inference complete.")


class Prep(Command):
    """🛠️ Prepare Steinegger Lab AFDB data into TFRecords for training."""
    
    def setup_arguments(self):
        opts = self.parser.add_argument_group("Inputs")
        opts.add_argument("db_path", type=str, help="Path prefix to Foldseek database (e.g. afdb50)")
        opts.add_argument("out_dir", type=str, help="Directory to output TFRecords")
        opts.add_argument("--num-workers", type=int, default=4, help="Number of parallel PyTorch DataLoader workers")
        opts.add_argument("--prefix", type=str, default=None, help="Prefix for the output TFRecord files (defaults to db-path basename)")
        
    def __call__(self, args: argparse.Namespace):
        self.cli.msg(f"🛠️ Initializing PyTorch DataLoader for Foldseek prep...")
        from foldgemma.data.prep import write_tfrecords_from_foldseek
        total = write_tfrecords_from_foldseek(args.db_path, args.out_dir, args.num_workers, args.prefix)
        self.cli.msg(f"✅ Data prep complete! Successfully serialized {total} records to TFRecords.")


class Deploy(Command):
    """🚀 Deploy a trained model to the Hugging Face Hub."""

    def setup_arguments(self):
        opts = self.parser.add_argument_group("Inputs")
        opts.add_argument("--repo-id", type=str, required=True, help="Target Hugging Face repository ID (e.g. username/foldgemma)")
        opts.add_argument("--model-path", type=str, default="./model.safetensors", help="Path to the model file")
        opts.add_argument("--token", type=str, default=None, help="HF API token. Falls back to HF_TOKEN env var if not set.")

    def __call__(self, args: argparse.Namespace):
        self.cli.msg(f"🚀 Deploying {args.model_path} to {args.repo_id}...")
        from foldgemma.deploy import deploy_to_huggingface
        
        try:
            deploy_to_huggingface(repo_id=args.repo_id, model_path=args.model_path, token=args.token)
            self.cli.msg("✅ Deployment complete!")
        except Exception as e:
            self.cli.msg(f"❌ Deployment failed: {e}")
            raise


def main():
    with Cli(description="FoldGemma CLI") as cli:
        cli.add_command(Train())
        cli.add_command(Infer())
        cli.add_command(Prep())
        cli.add_command(Deploy())
        cli.run()

if __name__ == "__main__":
    main()
