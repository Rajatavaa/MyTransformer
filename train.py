import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset,random_split
from model import transformer_work
from config import get_config,get_weights_file_path

from dataset import BilingualDataset
from datasets import load_dataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path

def get_all_sentences(ds,lang):
    for items in ds:
        yield items[lang]

def get_tokenizer_load(config,ds,lang):
    tokenizer_path = Path(config["tokenizer.json"].format(lang))
    if not Path.exists(tokenizer_path):
        # WordLevel requires a vocab dict; provide an empty dict to be populated by the trainer
        tokenizer = Tokenizer(WordLevel({}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()  # type: ignore
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer    

def get_ds(config):
    ds_raw: HFDataset = load_dataset('csebuetnlp/BanglaNMT', f"{config['lang_src']}-{config['lang_tgt']}", split='train')  # type: ignore
    
    tokenizer_src = get_tokenizer_load(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_tokenizer_load(config, ds_raw, config['lang_tgt'])
    
    split_ds = ds_raw.train_test_split(test_size=0.1, seed=42)
    train_ds_raw = split_ds['train']
    val_ds_raw = split_ds['test']
    
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

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device{device}")
    Path(config['model_folder']).mkdir(parents = True,exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps = 1e-9)
    
    initital_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename =  get_weights_file_path(config,config['preload'])
        print(f"Preloading model{model_filename}")
        state = torch.load(model_filename)
        initital_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    loss_fn = nn.CrossEntropyLoss()