import onnxruntime

# 加载 ONNX 模型
session = onnxruntime.InferenceSession("resnet18.onnx")

# 获取模型的输入和输出名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 准备输入数据
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 进行推理
result = session.run([output_name], {input_name: x})
