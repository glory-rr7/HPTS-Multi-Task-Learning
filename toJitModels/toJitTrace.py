from config.config_utils import LoadConfig
from toJitModels.evalStructureModels import Release,Release_lightly,Release_lightly_extra,Custom
import torch

# 根据协议加载模型
config = LoadConfig()

model_name = config['run']['model']
model = model_name
if model == 'Release':
    model = Release.ResNet_UNet()
elif model == 'Release_lightly':
    model = Release_lightly.ResNet_UNet()
elif model == 'Release_lightly_extra':
    model = Release_lightly_extra.ResNet_UNet()
elif model == 'Custom':
    model = Custom.ResNet_UNet(base =config['custom']['base'],
                                    refinement =config['custom']['refinement'],
                                    ffp =config['custom']['ffp'],
                                    ppm =config['custom']['ppm'],
                                    down_sample =config['custom']['down_sample'],
                                    am =config['custom']['am'])
else:
    print("Error: Model is not exist!")
    exit(1)


# TODO 根据需要在这个位置输入保存的模型权重
model_path = "./8.pth"
model.load_state_dict(torch.load(model_path))
model.to('cpu')
model.eval()

# # 固定权重
# for module in model.modules():
#     if hasattr(module, 'weight_orig'):
#         torch.nn.utils.remove_spectral_norm(module)


# 导出trace
dummy = torch.randn(1, 3, 512,512, device="cpu")
script = torch.jit.trace(model, dummy)
#print(script)
output = script(dummy)
print(model.get_name())
torch.jit.save(script, f'./{model.get_name()}.pt')
