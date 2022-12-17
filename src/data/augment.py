'''
Different NLP augmentation strategies
'''
import os
import json
from textattack.augmentation import EasyDataAugmenter
from tqdm import tqdm 

class Augmenter():
    def __init__(self, data_name, cache_dir, method='eda', aug_num=3, change_amount=0.9):
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.aug_num = aug_num
        self.change_amount = change_amount
        self.method = method
        
        method_dict = {
            'eda'   :   self._eda
        }
        self.method_func = method_dict[method]
    
    def augment(self, source_data, train=True):
        # check to see if augmented data is already saved
        fname = f'{self.cache_dir}/augmented/train{train}_{self.data_name}_{self.method}_aug-num{self.aug_num}_change{self.change_amount}.json'
        if not os.path.isdir(f'{self.cache_dir}/augmented'):
            os.mkdir(f'{self.cache_dir}/augmented')
        try:
            with open(fname, 'r') as f:
                aug_data = json.loads(f.read())
        except:
            print("First Time augmenting this data")
            aug_data = self.method_func(source_data, self.aug_num, self.change_amount)
            # save the augmented data
            with open(fname, 'w') as f:
                json.dump(aug_data, f)
        return aug_data
    
    @staticmethod
    def _eda(source_data, aug_num=3, change_frac=0.9):
        eda_aug = EasyDataAugmenter(transformations_per_example=aug_num, pct_words_to_swap=change_frac)
        aug_data = []
        for d in tqdm(source_data):
            aug_texts = eda_aug.augment(d['text'])
            aug_data += [{'text':t, 'label':d['label']} for t in aug_texts]
        return source_data + aug_data