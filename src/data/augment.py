'''
Different NLP augmentation strategies
'''
import os
import json
import torch
from textattack.augmentation import EasyDataAugmenter
import nlpaug.augmenter.word as naw
from tqdm import tqdm 

class Augmenter():
    def __init__(self, data_name, cache_dir, method='eda', aug_num=3, change_amount=0.9):
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.aug_num = aug_num
        self.change_amount = change_amount
        self.method = method
        
        method_dict = {
            'eda'   :   self._eda,
            'bt'    :   self._bt
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
            aug_data = self.method_func(source_data, aug_num=self.aug_num, change_amount=self.change_amount)
            # save the augmented data
            with open(fname, 'w') as f:
                json.dump(aug_data, f)
        return aug_data
    
    @staticmethod
    def _eda(source_data, aug_num=3, change_amount=0.9, *args, **kwargs):
        eda_aug = EasyDataAugmenter(transformations_per_example=aug_num, pct_words_to_swap=change_amount)
        aug_data = []
        for d in tqdm(source_data):
            aug_texts = eda_aug.augment(d['text'])
            aug_data += [{'text':t, 'label':d['label']} for t in aug_texts]
        return source_data + aug_data
    
    @staticmethod
    def _bt(source_data, aug_num=3, *args, **kwargs):
        # Back translation - will assume aug_num=3 for now
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        
        # en->fr->ru->es->en
        aug1a = naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-en-fr',
            to_model_name = f'Helsinki-NLP/opus-mt-fr-ru',
            device=device)
        aug1b = naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-ru-es',
            to_model_name = f'Helsinki-NLP/opus-mt-es-en',
            device=device)

        # en->es->de->fr->en
        aug2a = naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-en-es',
            to_model_name = f'Helsinki-NLP/opus-mt-es-de',
            device=device)
        aug2b = naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-de-fr',
            to_model_name = f'Helsinki-NLP/opus-mt-fr-en',
            device=device)

        # en->de->es->ru->en
        aug3a = naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-en-de',
            to_model_name = f'Helsinki-NLP/opus-mt-de-es',
            device=device)
        aug3b = naw.BackTranslationAug(
            from_model_name=f'Helsinki-NLP/opus-mt-es-ru',
            to_model_name = f'Helsinki-NLP/opus-mt-ru-en',
            device=device)

        
        aug_data = []
        for d in tqdm(source_data):
            aug_text = aug1b.augment(aug1a.augment(d['text'])[0])[0]
            aug_data.append({'text':aug_text, 'label':d['label']})

            aug_text = aug2b.augment(aug2a.augment(d['text'])[0])[0]
            aug_data.append({'text':aug_text, 'label':d['label']})

            aug_text = aug3b.augment(aug3a.augment(d['text'])[0])[0]
            aug_data.append({'text':aug_text, 'label':d['label']})
        return source_data + aug_data
