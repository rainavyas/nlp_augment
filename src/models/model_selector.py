from .models import SequenceClassifier
import torch

def select_model(model_name='bert-base-uncased', model_path=None, pretrained=True):
    model =  SequenceClassifier(model_name=model_name, pretrained=pretrained)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    return model