import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from model import transformer_work
from config import get_config
from dataset import causal_mask


def diagnose_model():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(f"tokenizer{config['lang_src']}.json")
    tokenizer_tgt = Tokenizer.from_file(f"tokenizer{config['lang_tgt']}.json")

    # Load model
    model = transformer_work(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    ).to(device)

    # Find latest checkpoint
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    checkpoint_dir = Path(model_folder)

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
        except:
            continue

    if not latest_checkpoint:
        print("No checkpoint found!")
        return

    print(f"Loading checkpoint: {latest_checkpoint}")
    state = torch.load(latest_checkpoint, map_location=device)

    # Handle torch.compile() state dict from older checkpoints (has '_orig_mod.' prefix)
    state_dict = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("Removing '_orig_mod.' prefix from torch.compile() checkpoint...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"\nLoaded model from epoch {state.get('epoch')}, global step {state.get('global_step')}"
    )
    print(f"Device: {device}")

    # Test sentence
    test_sentence = "Hello, how are you?"
    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC TEST")
    print(f"{'=' * 70}")
    print(f"Input: {test_sentence}")

    # Tokenize
    src_tokens = tokenizer_src.encode(test_sentence).ids
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx_src = tokenizer_src.token_to_id("[EOS]")
    pad_idx = tokenizer_src.token_to_id("[PAD]")

    enc_num_padding = config["seq_len"] - len(src_tokens) - 2
    encoder_input = (
        torch.cat(
            [
                torch.tensor([sos_idx], dtype=torch.int64),
                torch.tensor(src_tokens, dtype=torch.int64),
                torch.tensor([eos_idx_src], dtype=torch.int64),
                torch.tensor([pad_idx] * enc_num_padding, dtype=torch.int64),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )

    encoder_mask = (encoder_input != pad_idx).unsqueeze(1).unsqueeze(1).int().to(device)

    # Get special tokens for target
    sos_idx_tgt = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx_tgt = tokenizer_tgt.token_to_id("[EOS]")
    pad_idx_tgt = tokenizer_tgt.token_to_id("[PAD]")

    print(f"\nSpecial token IDs:")
    print(f"  Target [SOS]: {sos_idx_tgt}")
    print(f"  Target [EOS]: {eos_idx_tgt}")
    print(f"  Target [PAD]: {pad_idx_tgt}")

    # Encode
    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)

        # Start with [SOS]
        decoder_input = (
            torch.empty(1, 1)
            .fill_(sos_idx_tgt)
            .type_as(encoder_input)
            .to(device)
            .long()
        )

        print(f"\n{'=' * 70}")
        print(f"STEP-BY-STEP DECODING (first 10 steps)")
        print(f"{'=' * 70}")

        for step in range(10):
            decoder_mask = (
                causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
            )
            out = model.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )
            logits = model.project(out[:, -1])  # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

            print(f"\nStep {step + 1}:")
            print(f"  Current sequence length: {decoder_input.size(1)}")
            print(f"  Top 5 predictions:")
            for i in range(5):
                token_id = top5_indices[0, i].item()
                prob = top5_probs[0, i].item()
                token = (
                    tokenizer_tgt.decode([token_id])
                    if token_id < tokenizer_tgt.get_vocab_size()
                    else f"ID:{token_id}"
                )
                special = ""
                if token_id == sos_idx_tgt:
                    special = " [SOS]"
                elif token_id == eos_idx_tgt:
                    special = " [EOS]"
                elif token_id == pad_idx_tgt:
                    special = " [PAD]"
                print(
                    f"    {i + 1}. Token {token_id:5d} (prob: {prob:.4f}): '{token}'{special}"
                )

            # Take argmax
            _, next_word = torch.max(logits, dim=1)
            next_token_id = next_word.item()

            print(
                f"  Selected token: {next_token_id} ({'[EOS]' if next_token_id == eos_idx_tgt else 'token'})"
            )

            # Add to sequence
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(encoder_input)
                    .fill_(next_token_id)
                    .to(device)
                    .long(),
                ],
                dim=1,
            )

            # Check if EOS
            if next_token_id == eos_idx_tgt:
                print(f"\n  >>> EOS detected at step {step + 1}! Decoding stopped.")
                break

        # Decode final output
        final_tokens = decoder_input.squeeze(0).cpu().numpy()
        final_text = tokenizer_tgt.decode(final_tokens)

        print(f"\n{'=' * 70}")
        print(f"FINAL RESULT")
        print(f"{'=' * 70}")
        print(f"Token IDs: {final_tokens}")
        print(f"Decoded text: '{final_text}'")
        print(f"Text length: {len(final_text)}")

        # Check entropy of predictions
        decoder_input_first = (
            torch.empty(1, 1)
            .fill_(sos_idx_tgt)
            .type_as(encoder_input)
            .to(device)
            .long()
        )
        decoder_mask_first = causal_mask(1).type_as(encoder_mask).to(device)
        out_first = model.decode(
            decoder_input_first, encoder_output, encoder_mask, decoder_mask_first
        )
        logits_first = model.project(out_first[:, -1])
        probs_first = F.softmax(logits_first, dim=-1)
        entropy = -(probs_first * torch.log(probs_first + 1e-10)).sum().item()

        print(f"\nEntropy of first prediction: {entropy:.4f}")
        print(f"  (Low entropy < 1.0 suggests model is very confident/collapsed)")
        print(f"  (High entropy > 5.0 suggests model is uncertain)")


if __name__ == "__main__":
    diagnose_model()
