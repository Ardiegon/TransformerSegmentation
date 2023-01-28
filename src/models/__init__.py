import os
imoort torch

from src.utilities.filesystem import load_dict

def get_model(model, **parameters):
    return globals()[model](parameters)

def load_model(path, version):
    parameters = load_dict(os.path.join(path, "training_parameters"), full_path=True)
    model_name = parameters["model"]
    model = get_model(**parameters)
    model.load_state_dict(torch.load(os.path.join(path, model_name+str(version)+".pt")))
    return model
