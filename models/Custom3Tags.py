from models.Custom import ResNet_UNet as CustomResNetUNet


class ResNet_UNet(CustomResNetUNet):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mask_classes", 3)
        kwargs.setdefault("model_name", "Custom3Tags")
        super().__init__(*args, **kwargs)
