import os
import fire
from tqdm import tqdm
import random
# python -m data.imdb --save_path data/imdb --imdb_path data/imdb/aclImdb/
TEST_POS_DIR = 'test/pos'
TEST_NEG_DIR = 'test/neg'
TRAIN_POS_DIR = 'train/pos/'
TRAIN_NEG_DIR = 'train/neg/'
TRAIN_UNSUP_DIR = 'train/unsup/'

LM_TRAIN_FILE = 'lm_train.txt'
LM_VAL_FILE = 'lm_val.txt'


def process_data_for_language_model(save_path, imdb_path, val_sample=5000):
    """
        Read the IMDB folder files and create language model training and
        validation file

        :params:
            - save_path: Path where to save lm_train.txt and lm_valid.txt
            - imdb_path: Root directory of imdb dataset
            - val_sample: Number of files to select as validation set
    """
    pos = [open(os.path.join(imdb_path, TRAIN_POS_DIR, i)).read() for i in\
                            tqdm(os.listdir(os.path.join(imdb_path, TRAIN_POS_DIR)))]
    neg = [open(os.path.join(imdb_path, TRAIN_NEG_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TRAIN_NEG_DIR)))]
    test_pos = [open(os.path.join(imdb_path, TEST_POS_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TEST_POS_DIR)))]
    test_neg = [open(os.path.join(imdb_path, TEST_NEG_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TEST_NEG_DIR)))]
    unsup = [open(os.path.join(imdb_path, TRAIN_UNSUP_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TRAIN_UNSUP_DIR)))]

    text = pos + neg + unsup + test_neg + test_pos

    print('Test Data Creation..')
    val_index = random.sample(list(range(len(text))),val_sample)
    train = [i for idx, i in enumerate(text) if idx not in val_index]
    val = [i for idx, i in enumerate(text) if idx in val_index]
    open(os.path.join(save_path, LM_TRAIN_FILE), "w").write("\n".join(train))
    open(os.path.join(save_path, LM_VAL_FILE), "w").write("\n".join(val))


if __name__ == '__main__':
    fire.Fire(process_data_for_language_model)
