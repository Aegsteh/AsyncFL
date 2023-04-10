from model.CNN import CNN1, CNN3, VGG11s, VGG11, VGG11s_3, ResNet18
from model.LinearModel import logistic
def get_model(model_name):
    if model_name == 'CNN1':
        return CNN1()
    elif model_name == 'CNN3':
        return CNN3()
    elif model_name == 'VGG11s':
        return VGG11s()
    elif model_name == 'VGG11':
        return VGG11()
    elif model_name == 'VGG11s_3':
        return VGG11s_3()
    elif model_name == 'ResNet18':
        return ResNet18()
    elif model_name =='logistic':
        return logistic()