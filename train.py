import torch
import torch.nn as nn
from model import transformer_work

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path

def get_all_sentences(ds,lang):
    for items in ds:
        yield items['language'][lang]

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
    dataset = load_dataset('csebuetnlp/BanglaNMT',f'{config['lang_src']}-{config['lang_tgt']}',split='train')
    
    tokenizer_src = get_tokenizer_load(config,dataset,config['lang_src'])
    tokenizer_tgt = get_tokenizer_load(config,dataset,config['lang_tgt'])
    
def get_model(config,src_vocab_len,tgt_vocab_len):
    model = transformer_work(src_vocab_len,tgt_vocab_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

