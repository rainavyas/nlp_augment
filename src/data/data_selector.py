from .data_utils import load_data
from textattack.augmentation import EasyDataAugmenter
from tqdm import tqdm 

def select_data(args, train=True, aug=False, aug_num=3, change_frac=0.5):
    train_data, val_data, test_data = load_data(args.data_name, args.data_dir_path)

    if not train:
        return test_data
    
    if aug:
        # Augment all samples
        eda_aug = EasyDataAugmenter(transformations_per_example=aug_num, pct_words_to_swap=change_frac)
        train_data += augment(eda_aug, train_data)
        val_data += augment(eda_aug, val_data)
    return val_data, train_data

def augment(eda_aug, data):
    for d in tqdm(data):
        aug_texts = eda_aug.augment(d['text'])
        aug_data = [{'text':t, 'label':d['label']} for t in aug_texts]
    return aug_data


