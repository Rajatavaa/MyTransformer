import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset,random_split
import torchmetrics 
from model import transformer_work
from config import get_config,get_weights_file_path
from tqdm import tqdm
from warnings import filterwarnings

import os
from dataset import BilingualDataset,causal_mask
from datasets import load_dataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
import math

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device).long()
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device).long()], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def noam_scheduler(step,warmup_steps,d_model):
     return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device).long() # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds,lang):
    for items in ds:
        yield items[lang]

def get_tokenizer_load(config,ds,lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # WordLevel requires a vocab dict; provide an empty dict to be populated by the trainer
        tokenizer = Tokenizer(WordLevel({}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()  # type: ignore
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer    

def get_ds(config):
    # Load dataset from local JSONL files and limit to first 200k rows
    ds_raw: HFDataset = load_dataset('json', data_files='BanglaNMT/train.jsonl', split='train')  # type: ignore

    # Limit to first 200k rows
    original_size = len(ds_raw)
    if len(ds_raw) > 200000:
        ds_raw = ds_raw.select(range(200000))
        print(f'Dataset limited from {original_size} to {len(ds_raw)} rows')
    else:
        print(f'Dataset size: {len(ds_raw)} rows (no limiting needed)')

    tokenizer_src = get_tokenizer_load(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_tokenizer_load(config, ds_raw, config['lang_tgt'])

    # Use the limited dataset for training
    train_ds_raw = ds_raw
    val_ds_raw: HFDataset = load_dataset('json', data_files='BanglaNMT/validation.jsonl', split='train')  # type: ignore

    print(f'Training dataset: {len(train_ds_raw)} rows')
    print(f'Validation dataset: {len(val_ds_raw)} rows')

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
        
def get_model(config,src_vocab_len,tgt_vocab_len):
    model = transformer_work(src_vocab_len,tgt_vocab_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

def get_latest_checkpoint(config):
    """Find the latest checkpoint file based on global_step."""
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    checkpoint_dir = Path(model_folder)
    
    if not checkpoint_dir.exists():
        return None
    
    latest_checkpoint = None
    latest_step = -1
    
    # Look for all checkpoint files
    for checkpoint_file in checkpoint_dir.glob("*.pt"):
        try:
            state = torch.load(checkpoint_file, map_location='cpu')
            if 'global_step' in state:
                step = state['global_step']
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = checkpoint_file
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
            continue
    
    if latest_checkpoint:
        return str(latest_checkpoint)
    return None

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device{device}")
    # Create the correct model folder that matches get_weights_file_path
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    Path(model_folder).mkdir(parents = True,exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    model = torch.compile(model)
    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps = 1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: noam_scheduler(step + 1, config['warmup_steps'], config['d_model']))
    
    #The next steps help in retreiving back the model weights if any error happens during training
    initial_epoch = 0
    global_step = 0
    batches_to_skip = 0
    resume_state = None
    if config['preload']:
        model_filename =  get_weights_file_path(config,config['preload'])
        print(f"Preloading model{model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict']) #type:ignore
        resume_state = state
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print(f"Resumed from epoch {state['epoch']}, global step {global_step}")
        # If preload is True or "latest", find the latest checkpoint automatically
        if config['preload'] == True or config['preload'] == "latest":
            model_filename = get_latest_checkpoint(config)
            if model_filename:
                print(f"Auto-detected latest checkpoint: {model_filename}")
            else:
                print("Warning: No checkpoint found. Starting training from scratch.")
                model_filename = None
        else:
            # Use the specified checkpoint
            model_filename = get_weights_file_path(config, config['preload'])
        
        if model_filename and os.path.exists(model_filename):
            print(f"Preloading model: {model_filename}")
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            resume_state = state
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            print(f"Resumed from epoch {state['epoch']}, global step {global_step}")
        elif model_filename:
            print(f"Warning: Checkpoint file {model_filename} not found. Starting training from scratch.")
    
    steps_per_epoch = len(train_dataloader)

    if resume_state:
        if steps_per_epoch > 0:
            completed_batches = global_step % steps_per_epoch
        else:
            completed_batches = 0

        if completed_batches == 0 and global_step != 0:
            initial_epoch = resume_state['epoch'] + 1
        else:
            initial_epoch = resume_state['epoch']
            batches_to_skip = completed_batches

        if batches_to_skip:
            print(f"Skipping the first {batches_to_skip} batches of epoch {initial_epoch:02d} to resume mid-epoch.")
    else:
        initial_epoch = 0
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device) #label_smoothing allows to less overfit thus increading the accuracy and the 0.1 means taking 10% off of all the high probability tokens and distribute in others
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader,desc = f"Processing epoch{epoch:02d}")
        for batch_index, batch in enumerate(batch_iterator):
            if epoch == initial_epoch and batches_to_skip and batch_index < batches_to_skip:
                continue
            encoder_input = batch['encoder_input'].to(device).long() #(B,Seq_len)
            decoder_input = batch['decoder_input'].to(device).long() #(B,Seq_len)
            encoder_mask = batch['encoder_mask'].to(device)   #(B,1,1,Seq_len)
            decoder_mask = batch['decoder_mask'].to(device)   #(B,1,Seq_len,Seq_len)

            encoder_output = model.encode(encoder_input,encoder_mask) #(B,seq_len,d_model)
            decoder_output = model.decode(decoder_input,encoder_output,encoder_mask,decoder_mask) #(B,seq_len,d_model)
            proj_output = model.project(decoder_output) #(B,seq_len,vocab_size)
            label = batch['label'].to(device).long()
            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Save checkpoint every N batches (mid-epoch checkpointing)
            if global_step % config['checkpoint_interval'] == 0:
                checkpoint_filename = get_weights_file_path(config, f"checkpoint_step_{global_step}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'global_step': global_step
                }, checkpoint_filename)
                print(f"\nCheckpoint saved at step {global_step}: {checkpoint_filename}")

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    
    config = get_config()
    train_model(config)
            
            
            
            