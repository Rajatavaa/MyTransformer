from pathlib import Path

def get_config():
    return {
        "batch_size":10,
        "seq_len":512,
        "d_model":512,
        "num_epochs":3,
        "lr":10e-4,
        "warmup_steps":5000,
        "lang_src":"en",
        "lang_tgt":"bn",
        "datasource":"banglanlp_nmt",
        "save_folder":"weights",
        "model_folder": "weights",
        "experiment_name":"runs/tmodel",
        "tokenizer_file":"tokenizer{0}.json",
        "model_basename":"t_model" ,
        "preload":None,
        "checkpoint_interval": 5000  
    }
    
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
