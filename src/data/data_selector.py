from .data_utils import load_data
from textattack.augmentation import EasyDataAugmenter

def data_sel(args, train=True, aug=False):
    train_data, val_data, test_data = load_data(args.data_name, args.data_dir_path)

    if not train:
        return test_data
    
    if aug:
        # Augment all samples
        eda_aug = EasyDataAugmenter()

        aug_train_data = [{'text':eda_aug.augment(d['text']), 'label':d['label']} for d in train_data]
        aug_val_data = [{'text':eda_aug.augment(d['text']), 'label':d['label']} for d in train_data]

        train_data += aug_train_data
        val_data += aug_val_data
    
    return val_data, train_data


