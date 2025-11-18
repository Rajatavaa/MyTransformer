from pathlib import Path

def get_config():
    return {
        "batch_size":32,
        "seq_len":256,
        "d_model":512,
        "num_epochs":3,
        "lr":1,
        "warmup_steps":4000,
        "lang_src":"en",
        "lang_tgt":"bn",
        "datasource":"banglanlp_nmt",
        "save_folder":"weights",
        "model_folder": "weights",
        "experiment_name":"runs/tmodel",
        "tokenizer_file":"tokenizer{0}.json",
        "model_basename":"t_model" ,
        "preload":"latest",
        "checkpoint_interval": 1000  
    }
    
def get_weights_file_path(config, epoch: str):
    epoch_path = Path(epoch)

    # If an absolute path is provided, respect it as-is.
    if epoch_path.is_absolute():
        return str(epoch_path)

    model_folder = f"{config['datasource']}_{config['model_folder']}"

    # Ensure the filename includes the model basename and .pt suffix
    filename = epoch
    if not filename.startswith(config['model_basename']):
        filename = f"{config['model_basename']}{filename}"
    if not filename.endswith(".pt"):
        filename = f"{filename}.pt"

    return str(Path('.') / model_folder / filename)
