# model_loader.py
# model 추가 및 변경시 수정

import torch
from network.models import model_selection

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == 'xception':
            model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
            model_path = 'model/deepfake_c0_xception.pkl'

        elif self.model_name == '1104':
            model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
            model_path = 'model/1104배포.pth'
        elif self.model_name == 'best_model_fold_5.pkl':
            model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
            model_path = 'model/best_model_fold_5.pkl'
        else:
            raise ValueError(f"Model '{self.model_name}' not recognized.")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.to(self.device)
        model.eval()
        return model
