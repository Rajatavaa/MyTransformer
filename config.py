def get_config():
    return {
        "batch_size":8,
        "seq_len":350,
        "d_model":512,
        "num_epochs":20,
        "lr":"10e-4",
        "lang_src":"en",
        "lang_tgt":"be",
        "save_folder":"weights",
        "experiment_name":"t_model",
        "tokenizer_file":"tokenizer{0}.json"    
    }
    
def get_