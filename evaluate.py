import torch
import argparse
from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm

from model import transformer_work
from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """Greedy decoding for translation."""
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device).long()

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(source)
                .fill_(next_word.item())
                .to(device)
                .long(),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def load_tokenizers(config):
    """Load source and target tokenizers."""
    tokenizer_src_path = Path(config["tokenizer_file"].format(config["lang_src"]))
    tokenizer_tgt_path = Path(config["tokenizer_file"].format(config["lang_tgt"]))

    if not tokenizer_src_path.exists():
        raise FileNotFoundError(f"Source tokenizer not found: {tokenizer_src_path}")
    if not tokenizer_tgt_path.exists():
        raise FileNotFoundError(f"Target tokenizer not found: {tokenizer_tgt_path}")

    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    return tokenizer_src, tokenizer_tgt


def load_model_from_checkpoint(
    checkpoint_path, config, tokenizer_src, tokenizer_tgt, device
):
    """Load model from a checkpoint file."""
    model = transformer_work(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    # Handle torch.compile() state dict from older checkpoints (has '_orig_mod.' prefix)
    state_dict = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("Removing '_orig_mod.' prefix from torch.compile() checkpoint...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    epoch = state.get("epoch", "unknown")
    global_step = state.get("global_step", "unknown")
    print(f"Loaded model from epoch {epoch}, global step {global_step}")

    return model


def get_latest_checkpoint(config):
    """Find the latest checkpoint file based on global_step."""
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    checkpoint_dir = Path(model_folder)

    if not checkpoint_dir.exists():
        return None

    latest_checkpoint = None
    latest_step = -1

    for checkpoint_file in checkpoint_dir.glob("*.pt"):
        try:
            state = torch.load(checkpoint_file, map_location="cpu")
            if "global_step" in state:
                step = state["global_step"]
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = checkpoint_file
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
            continue

    if latest_checkpoint:
        return str(latest_checkpoint)
    return None


def translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, config, device):
    """Translate a single sentence."""
    model.eval()

    # Tokenize the source sentence
    src_tokens = tokenizer_src.encode(sentence).ids

    # Truncate if too long
    if len(src_tokens) > config["seq_len"] - 2:
        src_tokens = src_tokens[: config["seq_len"] - 2]

    # Build encoder input with SOS, EOS, and padding
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")
    pad_idx = tokenizer_src.token_to_id("[PAD]")

    enc_num_padding = config["seq_len"] - len(src_tokens) - 2
    encoder_input = (
        torch.cat(
            [
                torch.tensor([sos_idx], dtype=torch.int64),
                torch.tensor(src_tokens, dtype=torch.int64),
                torch.tensor([eos_idx], dtype=torch.int64),
                torch.tensor([pad_idx] * enc_num_padding, dtype=torch.int64),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )

    # Create encoder mask
    encoder_mask = (encoder_input != pad_idx).unsqueeze(1).unsqueeze(1).int().to(device)

    with torch.no_grad():
        model_out = greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
        )

    # Decode the output
    translation = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    return translation


def evaluate_on_dataset(
    model, dataloader, tokenizer_src, tokenizer_tgt, config, device, num_samples=None
):
    """Evaluate model on a dataset and compute metrics."""
    model.eval()

    source_texts = []
    expected = []
    predicted = []

    count = 0
    total = num_samples if num_samples else len(dataloader)

    print(f"\nEvaluating on {total} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, total=total, desc="Evaluating"):
            encoder_input = batch["encoder_input"].to(device).long()
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                config["seq_len"],
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            count += 1
            if num_samples and count >= num_samples:
                break

    # Compute metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Character Error Rate
    cer_metric = torchmetrics.CharErrorRate()
    cer = cer_metric(predicted, expected)
    print(f"Character Error Rate (CER): {cer:.4f}")

    # Word Error Rate
    wer_metric = torchmetrics.WordErrorRate()
    wer = wer_metric(predicted, expected)
    print(f"Word Error Rate (WER): {wer:.4f}")

    # BLEU Score
    bleu_metric = torchmetrics.BLEUScore()
    bleu = bleu_metric(predicted, expected)
    print(f"BLEU Score: {bleu:.4f}")

    print("=" * 60)

    return {
        "cer": cer.item(),
        "wer": wer.item(),
        "bleu": bleu.item(),
        "source_texts": source_texts,
        "expected": expected,
        "predicted": predicted,
    }


def show_examples(results, num_examples=5):
    """Display translation examples."""
    print(f"\n{'=' * 60}")
    print(f"SAMPLE TRANSLATIONS (showing {num_examples} examples)")
    print("=" * 60)

    for i in range(min(num_examples, len(results["source_texts"]))):
        print(f"\n[Example {i + 1}]")
        print(f"  SOURCE:    {results['source_texts'][i]}")
        print(f"  EXPECTED:  {results['expected'][i]}")
        print(f"  PREDICTED: {results['predicted'][i]}")
    print()


def interactive_mode(model, tokenizer_src, tokenizer_tgt, config, device):
    """Interactive translation mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TRANSLATION MODE")
    print(
        f"Translating from {config['lang_src'].upper()} to {config['lang_tgt'].upper()}"
    )
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            sentence = input(f"Enter {config['lang_src'].upper()} sentence: ").strip()
            if sentence.lower() in ["quit", "exit", "q"]:
                print("Exiting interactive mode.")
                break
            if not sentence:
                continue

            translation = translate_sentence(
                model, sentence, tokenizer_src, tokenizer_tgt, config, device
            )
            print(f"Translation: {translation}\n")

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Transformer model from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: latest checkpoint)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of examples to display (default: 10)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Enable interactive translation mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available)",
    )

    args = parser.parse_args()

    # Setup
    config = get_config()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load tokenizers
    print("Loading tokenizers...")
    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    print(f"Source vocab size: {tokenizer_src.get_vocab_size()}")
    print(f"Target vocab size: {tokenizer_tgt.get_vocab_size()}")

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        path_obj = Path(checkpoint_path)

        # Try multiple resolution strategies
        if path_obj.exists():
            # Path exists as-is (absolute or relative)
            checkpoint_path = str(path_obj)
        elif path_obj.is_absolute():
            # Absolute path provided but doesn't exist
            print(f"Error: Checkpoint not found at absolute path: {checkpoint_path}")
            return
        else:
            # Try as relative path from current directory
            if not path_obj.exists():
                # Try as filename in model folder
                model_folder = f"{config['datasource']}_{config['model_folder']}"
                filename_in_folder = Path(model_folder) / checkpoint_path
                if filename_in_folder.exists():
                    checkpoint_path = str(filename_in_folder)
                else:
                    # Try using get_weights_file_path (for epoch-style names)
                    resolved_path = get_weights_file_path(config, args.checkpoint)
                    if Path(resolved_path).exists():
                        checkpoint_path = resolved_path
                    else:
                        print(f"Error: Could not find checkpoint '{args.checkpoint}'")
                        print("Tried:")
                        print(f"  1. As-is: {path_obj}")
                        print(f"  2. In model folder: {filename_in_folder}")
                        print(f"  3. Resolved path: {resolved_path}")
                        print("\nAvailable checkpoints:")
                        checkpoint_dir = Path(model_folder)
                        if checkpoint_dir.exists():
                            for f in sorted(checkpoint_dir.glob("*.pt")):
                                print(f"  - {f.name}")
                        return
    else:
        checkpoint_path = get_latest_checkpoint(config)

    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        model_folder = f"{config['datasource']}_{config['model_folder']}"
        checkpoint_dir = Path(model_folder)
        if checkpoint_dir.exists():
            for f in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {f.name}")
        return

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path, config, tokenizer_src, tokenizer_tgt, device
    )
    model.eval()

    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer_src, tokenizer_tgt, config, device)
    else:
        # Evaluation mode
        print("\nLoading validation dataset...")
        val_ds_raw: HFDataset = load_dataset(
            "json", data_files="BanglaNMT/validation.jsonl", split="train"
        )  # type: ignore
        val_ds = BilingualDataset(
            val_ds_raw,
            tokenizer_src,
            tokenizer_tgt,
            config["lang_src"],
            config["lang_tgt"],
            config["seq_len"],
        )
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
        print(f"Validation dataset size: {len(val_ds)}")

        # Run evaluation
        results = evaluate_on_dataset(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config,
            device,
            args.num_samples,
        )

        # Show examples
        if args.show_examples > 0:
            show_examples(results, args.show_examples)


if __name__ == "__main__":
    main()
