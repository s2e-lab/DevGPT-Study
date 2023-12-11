import torch.onnx
import torchvision.models as models

# 创建一个预训练的模型
model = models.resnet18(pretrained=True)

# 设置模型为评估模式
model.eval()

# 定义一个输入张量
x = torch.randn(1, 3, 224, 224)

# 导出模型
torch.onnx.export(model, x, "resnet18.onnx")
