import math
import argparse
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.text import CharErrorRate, WordErrorRate

from model import transformer_work
from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask


# =========================

# TEXT UTILITIES

# =========================


def normalize_text(text):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.replace("\u0120", " ")
    text = text.replace("\u200d", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200b", "")
    text = "".join(
        ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf") or ch in ("\n", "\t")
    )
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text.strip()


def tokenize_bengali(text):
    return text.split()


# =========================

# BLEU METRICS

# =========================


def _count_ngrams(tokens, max_n):
    ngram_counts = Counter()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram_counts[tuple(tokens[i : i + n])] += 1
    return ngram_counts


def corpus_bleu(preds, refs, max_n=4):
    total_clip = [0] * max_n
    total_count = [0] * max_n
    ref_len_total = 0
    pred_len_total = 0

    for pred, ref_list in zip(preds, refs):
        pred_tokens = pred.split()
        ref_token_lists = [r.split() for r in ref_list]
        ref_lens = [len(rt) for rt in ref_token_lists]

        closest_idx = min(range(len(ref_lens)), key=lambda i: abs(ref_lens[i] - len(pred_tokens)))
        ref_len_total += ref_lens[closest_idx]
        pred_len_total += len(pred_tokens)

        pred_ngrams = _count_ngrams(pred_tokens, max_n)
        max_ref_ngrams = Counter()
        for ref_tokens in ref_token_lists:
            ref_ngrams = _count_ngrams(ref_tokens, max_n)
            for ngram, count in ref_ngrams.items():
                max_ref_ngrams[ngram] = max(max_ref_ngrams.get(ngram, 0), count)

        for ngram, count in pred_ngrams.items():
            n = len(ngram) - 1
            if n < max_n:
                total_clip[n] += min(count, max_ref_ngrams.get(ngram, 0))
                total_count[n] += count

    if pred_len_total == 0:
        return 0.0

    if pred_len_total < ref_len_total:
        bp = math.exp(1 - ref_len_total / pred_len_total)
    else:
        bp = 1.0

    log_avg = 0.0
    for n in range(max_n):
        if total_count[n] == 0:
            return 0.0
        precision = (total_clip[n] + 1) / (total_count[n] + 1)
        log_avg += math.log(precision)
    log_avg /= max_n

    return bp * math.exp(log_avg) * 100


def _get_char_ngrams(text, min_n=3, max_n=6):
    words = text.split()
    ngrams = []
    for word in words:
        chars = list(word)
        for n in range(min_n, max_n + 1):
            for i in range(len(chars) - n + 1):
                ngrams.append(tuple(chars[i : i + n]))
    return ngrams


def char_bleu(preds, refs, min_n=3, max_n=6):
    num_levels = max_n - min_n + 1
    total_clip = [0] * num_levels
    total_count = [0] * num_levels
    ref_len_total = 0
    pred_len_total = 0

    for pred, ref_list in zip(preds, refs):
        pred_char_lens = sum(len(w) for w in pred.split())
        pred_len_total += pred_char_lens

        best_clip = [Counter() for _ in range(num_levels)]
        best_count = [Counter() for _ in range(num_levels)]

        best_ref_len = float("inf")
        for ref in ref_list:
            ref_char_lens = sum(len(w) for w in ref.split())
            if abs(ref_char_lens - pred_char_lens) < abs(best_ref_len - pred_char_lens):
                best_ref_len = ref_char_lens

            ref_ngrams = Counter(_get_char_ngrams(ref, min_n, max_n))
            pred_ngrams = Counter(_get_char_ngrams(pred, min_n, max_n))
            clipped = pred_ngrams & ref_ngrams

            for ngram, count in clipped.items():
                idx = len(ngram) - min_n
                if 0 <= idx < num_levels:
                    best_clip[idx][ngram] = max(best_clip[idx].get(ngram, 0), count)

            for ngram, count in pred_ngrams.items():
                idx = len(ngram) - min_n
                if 0 <= idx < num_levels:
                    best_count[idx][ngram] = max(best_count[idx].get(ngram, 0), count)

        ref_len_total += best_ref_len

        for idx in range(num_levels):
            total_clip[idx] += sum(best_clip[idx].values())
            total_count[idx] += sum(best_count[idx].values())

    if pred_len_total == 0:
        return 0.0

    if pred_len_total < ref_len_total:
        bp = math.exp(1 - ref_len_total / pred_len_total)
    else:
        bp = 1.0

    if any(c == 0 for c in total_count):
        return 0.0

    log_avg = 0.0
    for idx in range(num_levels):
        precision = (total_clip[idx] + 1) / (total_count[idx] + 1)
        log_avg += math.log(precision)
    log_avg /= num_levels

    return bp * math.exp(log_avg) * 100


# =========================

# GREEDY DECODING

# =========================


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        logits = model.project(out[:, -1])

        probs = torch.softmax(logits, dim=-1)
        next_word = torch.argmax(probs, dim=-1)

        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


# =========================

# BEAM SEARCH

# =========================


def length_penalty(seq_len, alpha=0.6):
    return ((5 + seq_len) / (5 + 1)) ** alpha


def get_blocked_tokens(tokens, n=3):
    if len(tokens) < n:
        return set()

    seen = {}
    blocked = set()

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n - 1])
        next_token = tokens[i + n - 1]

        if ngram in seen:
            blocked.add(next_token)
        seen[ngram] = True

    return blocked


def beam_search_decode(
    model,
    source,
    source_mask,
    tokenizer_tgt,
    max_len,
    device,
    beam_size=5,
    alpha=0.6,
    ngram_block=3,
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    pad_idx = tokenizer_tgt.token_to_id("[PAD]")

    encoder_output = model.encode(source, source_mask)

    beams = [
        {
            "tokens": [sos_idx],
            "logp": 0.0,
            "finished": False,
        }
    ]

    for _ in range(max_len - 1):
        all_candidates = []

        for beam in beams:
            if beam["finished"]:
                all_candidates.append(beam)
                continue

            seq_len = len(beam["tokens"])
            decoder_input = torch.tensor([beam["tokens"]], dtype=torch.long).to(device)
            decoder_mask = causal_mask(seq_len).to(device)

            out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
            logits = model.project(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            blocked_ids = get_blocked_tokens(beam["tokens"], n=ngram_block)
            for bid in blocked_ids:
                log_probs[bid] = float("-inf")
            log_probs[pad_idx] = float("-inf")

            topk_logp, topk_idx = log_probs.topk(beam_size)

            for j in range(beam_size):
                token_id = topk_idx[j].item()
                token_logp = topk_logp[j].item()

                new_tokens = beam["tokens"] + [token_id]
                new_logp = beam["logp"] + token_logp
                finished = token_id == eos_idx

                lp = length_penalty(len(new_tokens), alpha)
                score = new_logp / lp

                all_candidates.append(
                    {
                        "tokens": new_tokens,
                        "logp": new_logp,
                        "score": score,
                        "finished": finished,
                    }
                )

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        beams = all_candidates[:beam_size]

        if all(b["finished"] for b in beams):
            break

    best = max(beams, key=lambda x: x["score"])
    tokens = best["tokens"]
    if tokens[-1] == eos_idx:
        tokens = tokens[:-1]
    return torch.tensor(tokens[1:], dtype=torch.long)


# =========================

# DECODE HELPERS

# =========================


def decode_tokens(token_ids, tokenizer_tgt):
    ids = token_ids.cpu().numpy().tolist()

    cleaned = []
    for tid in ids:
        if tid in (
            tokenizer_tgt.token_to_id("[SOS]"),
            tokenizer_tgt.token_to_id("[EOS]"),
            tokenizer_tgt.token_to_id("[PAD]"),
        ):
            continue
        cleaned.append(tid)

    if not cleaned:
        return ""

    text = tokenizer_tgt.decode(cleaned, skip_special_tokens=True)
    return normalize_text(text)


# =========================

# LOAD TOKENIZERS

# =========================


def load_tokenizers(config):
    tokenizer_src = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_src"])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_tgt"])))
    )

    tokenizer_tgt.decoder = ByteLevelDecoder()

    return tokenizer_src, tokenizer_tgt


# =========================

# LOAD MODEL

# =========================


def load_model(checkpoint_path, config, tokenizer_src, tokenizer_tgt, device):
    model = transformer_work(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model_state_dict"]

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# =========================

# EVALUATION

# =========================


def evaluate(
    model,
    dataloader,
    tokenizer_tgt,
    config,
    device,
    num_samples=None,
    beam_size=5,
    alpha=0.6,
):
    model.eval()

    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()

    source_texts, expected, predicted = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = beam_search_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_tgt,
                config["seq_len"],
                device,
                beam_size=beam_size,
                alpha=alpha,
            )

            pred_text = decode_tokens(model_out, tokenizer_tgt)
            exp_text = normalize_text(batch["tgt_text"][0])

            source_texts.append(batch["src_text"][0])
            expected.append(exp_text)
            predicted.append(pred_text)

            if num_samples and i >= num_samples:
                break

    cer = cer_metric(predicted, expected)
    wer = wer_metric(predicted, expected)
    bleu = corpus_bleu(predicted, [[e] for e in expected])
    cbleu = char_bleu(predicted, [[e] for e in expected])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"CER:       {cer:.4f}")
    print(f"WER:       {wer:.4f}")
    print(f"BLEU:      {bleu:.2f}")
    print(f"char-BLEU: {cbleu:.2f}")
    print("=" * 60)

    return source_texts, expected, predicted


# =========================

# SHOW EXAMPLES

# =========================


def show_examples(src, exp, pred, n=5):
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)

    for i in range(min(n, len(src))):
        print(f"\n[Example {i + 1}]")
        print(f"SOURCE:    {src[i]}")
        print(f"EXPECTED:  {exp[i]}")
        print(f"PREDICTED: {pred[i]}")
        if i < 2:
            print(f"DEBUG PRED: {repr(pred[i][:120])}")
            print(f"DEBUG EXP:  {repr(exp[i][:120])}")


# =========================

# MAIN

# =========================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_src, tokenizer_tgt = load_tokenizers(config)

    model = load_model(args.checkpoint, config, tokenizer_src, tokenizer_tgt, device)

    val_ds_raw: HFDataset = load_dataset(
        "json", data_files="BanglaNMT/validation.jsonl", split="train"
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    val_loader = DataLoader(val_ds, batch_size=1)

    src, exp, pred = evaluate(
        model,
        val_loader,
        tokenizer_tgt,
        config,
        device,
        args.num_samples,
        beam_size=args.beam_size,
        alpha=args.alpha,
    )

    show_examples(src, exp, pred)


if __name__ == "__main__":
    main()
