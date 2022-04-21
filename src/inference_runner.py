import os
import copy
import pickle as pkl
from torch.utils.data import DataLoader
import transformers
from bpemb import BPEmb
import logging

from configuration import BaseConfig
from data_loader import read_json5, read_text
from models.mt5_transformer import Classifier
from evaluation import evaluate_with_seqeval

from data_preparation import prepare_conll_data, create_test_samples, pad_sequence_2
from utils import find_max_length_in_list, handle_subtoken_labels, convert_x_label_to_true_label, \
    progress_bar, label_correction
from inference import Inference, InferenceDataset

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    # MODEL_PATH = read_json5(os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
    #                                      "b_model_path.json"))["best_model_path"]

    # load BPEmb
    BPEMB = BPEmb(model_file=CONFIG.bpemb_model_path,
                  emb_file=CONFIG.bpemb_vocab_path, dim=300)

    logging.debug("test file : {}".format(CONFIG.dev_data))

    MODEL_PATH = "../assets/saved_models/turkish/subtoken_check/checkpoints/QTag-epoch=51-val_loss=0.13.ckpt"

    MODEL = Classifier.load_from_checkpoint(MODEL_PATH, map_location="cuda:0")
    MODEL.eval().to("cuda:0")

    TOKENIZER = transformers.MT5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # load raw data
    RAW_DATA = read_text(path=os.path.join(CONFIG.processed_data_dir, CONFIG.dev_data))

    TOKENS, LABELS = prepare_conll_data(RAW_DATA)
    TOKEN_ = copy.copy(TOKENS)

    SENTENCES, SUBTOKEN_CHECKS, BPEMB_IDS, FLAIR_TOKENS = create_test_samples(TOKEN_, TOKENIZER, BPEMB)
    SEN_MAX_LENGTH = find_max_length_in_list(SENTENCES)

    INFER = Inference(MODEL, TOKENIZER)

    DATASET = InferenceDataset(texts=SENTENCES, subtoken_checks=SUBTOKEN_CHECKS, bpemb_ids=BPEMB_IDS,
                               tokenizer=TOKENIZER, max_length=SEN_MAX_LENGTH, tokens=FLAIR_TOKENS)

    DATALOADER = DataLoader(DATASET, batch_size=1,
                            shuffle=False, num_workers=4)

    FILE = open("tr.pred.conll", "w")

    PREDICTED_LABELS = []
    for i_batch, sample_batched in enumerate(DATALOADER):
        sample_batched["input_ids"] = sample_batched["input_ids"].to("cuda:0")
        sample_batched["subtoken_check"] = sample_batched["subtoken_check"].to("cuda:0")
        sample_batched["bpemb_ids"] = sample_batched["bpemb_ids"].to("cuda:0")
        sample_batched["position"] = sample_batched["position"].to("cuda:0")
        OUTPUT = INFER.predict(sample_batched)

        ENTITIES = INFER.convert_ids_to_entities(OUTPUT)

        ENTITIES = handle_subtoken_labels(ENTITIES, SUBTOKEN_CHECKS[i_batch])

        ENTITIES = convert_x_label_to_true_label(ENTITIES, "X")
        assert len(ENTITIES) == len(LABELS[i_batch]), f"{len(LABELS[i_batch])}, {len(ENTITIES)}"

        PREDICTED_LABELS.append(ENTITIES)
        # ENTITIES = label_correction(ENTITIES)

        for entity in ENTITIES:
            FILE.write(entity.strip())
            FILE.write("\n")
        FILE.write("\n")

        progress_bar(i_batch, len(DATALOADER), "testing ....")

    DATA = [TOKENS, LABELS, PREDICTED_LABELS]
    with open("preds.pkl", "wb") as file:
        pkl.dump(DATA, file)

    # print(evaluate_with_seqeval(LABELS, PREDICTED_LABELS))
