import os
import numpy as np

from configuration import BaseConfig
from data_loader import read_json5
from transformers import BertTokenizer
from models.mt5_model import Classifier
from utils import ignore_pad_index, convert_subtoken_to_token

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    TEXT = "it was republished by mit press in 1971 and is still in print ."

    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)
    INPUTS = TOKENIZER.encode_plus(
        text=TEXT,
        add_special_tokens=True,
        max_length=21,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt"
    )

    # print(INPUTS) input_ids attention_mask
    MODEL_PATH = read_json5(os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
                                         "b_model_path.json"))["best_model_path"]

    MODEL = Classifier.load_from_checkpoint(MODEL_PATH)
    MODEL.eval()
    IDX2TAG = MODEL.hparams["idx2tag"]
    TOKENS = [TOKENIZER.convert_ids_to_tokens(idx) for idx in INPUTS["input_ids"]][0]

    OUTPUTS = MODEL(INPUTS)
    OUTPUTS = np.argmax(OUTPUTS.detach().numpy(), axis=2)
    OUTPUTS = [IDX2TAG[item] for item in OUTPUTS[0]]

    TOKENS, OUTPUTS = ignore_pad_index([TOKENS], [OUTPUTS])
    TOKEN2TAG = dict()

    TOKENS, OUTPUTS = convert_subtoken_to_token(TOKENS[0], OUTPUTS[0])

    print(TOKENS[1:])
    print(OUTPUTS[1:])
