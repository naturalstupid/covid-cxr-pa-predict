def func_call(func_name):
    n1 = 2
    n2 = 3
    return func_name(n1,n2)
def sum(n1,n2):
    return n1+n2
def product(n1,n2):
    return n1*n2
print(func_call(sum),func_call(product))
def get_model(model_name):
    available_models={'VGG16':VGG16, 'MobileNetV2':MobileNetV2, 'DenseNet201':DenseNet201, 'ResNet101V2':ResNet101V2}
    default_model = 'DenseNet201'
    if model_name in available_models:
        model = available_models[model_name](input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    else:
        model = available_models[default_model](input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    return model
