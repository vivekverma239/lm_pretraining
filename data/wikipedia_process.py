import os
import fire
from tqdm import tqdm
import random
import pickle
from data_utils import get_tokenizer
from sklearn.model_selection import train_test_split
import json

# python -m data.imdb --save_path data/imdb --imdb_path data/imdb/aclImdb/
DATA_DIR = 'data/wikiextractor/output/'
save_path = 'data/'
LM_TRAIN_FILE = 'lm_wiki_train.txt'
LM_VAL_FILE = 'lm_wiki_val.txt'


def process_data_for_language_model(val_sample=5000, tokenizer=None):
    """
        Read the IMDB folder files and create language model training and
        validation file

        :params:
            - save_path: Path where to save lm_train.txt and lm_valid.txt
            - imdb_path: Root directory of imdb dataset
            - val_sample: Number of files to select as validation set
    """
    all_files = []
    for root, dirnames, filenames in os.walk(DATA_DIR):
        all_files.extend([os.path.join(root, filename) for filename in filenames])
    train_files, test_files = train_test_split(all_files, test_size=0.1)

    train_text = [json.loads(line)["text"] for i in train_files for line in open(i) ]
    test_text = [json.loads(line)["text"] for i in test_files for line in open(i) ]

    tokenizer = get_tokenizer(tokenizer)

    process = lambda x: [' '.join(tokenizer(i)) for i in tqdm(x)]
    test = process(test_text)
    train = process(train_text)


    open(os.path.join(save_path, LM_TRAIN_FILE), "w").write("\n".join(train))
    open(os.path.join(save_path, LM_VAL_FILE), "w").write("\n".join(test))



if __name__ == '__main__':
    fire.Fire(process_data_for_language_model)
