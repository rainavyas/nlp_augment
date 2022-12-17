from .data_utils import load_data
from .augment import Augmenter



def select_data(args, train=True, aug=False, aug_num=3, aug_method='eda', change_amount=0.5):
    train_data, val_data, test_data = load_data(args.data_name, args.data_dir_path)

    if not train:
        return test_data
    
    if aug:
        # Augment all samples
        augmenter = Augmenter(args.data_name, args.data_dir_path, method=aug_method, aug_num=aug_num, change_amount=change_amount)
        train_data = augmenter.augment(train_data, train=True)
        val_data = augmenter.augment(val_data, train=False)
    return val_data, train_data



