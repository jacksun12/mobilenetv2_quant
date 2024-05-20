import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn  # 导入 nn 模块 Import nn module
from torch.utils.data import DataLoader
from model import mobilenet_v2, InvertedResidual

# 超参数
batch_size = 100

# 数据预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# 加载数据集
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(num_classes=100).to(device)
model.load_state_dict(torch.load('mobilenetv2_cifar100.pth'))
model.eval()
'''
# 将模型前端的权重量化为 int8 Quantize the weights of the model's front layers to int8
def quantize_model_front(model, num_layers=3):
    layers_to_quantize = [model.features[i] for i in range(num_layers)]
    for layer in layers_to_quantize:
        if isinstance(layer, nn.Conv2d):
            quantized_weight = torch.quantize_per_tensor(layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
            layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
        elif isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Conv2d):
                    quantized_weight = torch.quantize_per_tensor(sub_layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                    sub_layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_model_front(model)  # 量化模型的前端 Quantize the front part of the model
'''
'''
# 将模型前端的权重量化为 int8 Quantize the weights of the model's front layers to int8
def quantize_model_front(model, num_layers=4):
    layers_to_quantize = [model.features[i] for i in range(num_layers)]
    for layer in layers_to_quantize:
        if isinstance(layer, nn.Conv2d):
            quantized_weight = torch.quantize_per_tensor(layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
            layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
        elif isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Conv2d):
                    quantized_weight = torch.quantize_per_tensor(sub_layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                    sub_layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_model_front(model)  # 量化模型的前端 Quantize the front part of the model
'''
'''
# 将模型所有卷积层的权重量化为 int8 Quantize the weights of all convolutional layers in the model to int8
def quantize_model_conv(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            quantized_weight = torch.quantize_per_tensor(layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
            layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_model_conv(model)  # 量化模型的所有卷积层 Quantize all convolutional layers of the model
'''
'''
# 仅量化分类器的权重量化为 int8 Quantize only the classifier's weights to int8
def quantize_classifier(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            quantized_weight = torch.quantize_per_tensor(layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
            layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_classifier(model)  # 仅量化分类器 Quantize only the classifier
'''
'''
# 量化每个InvertedResidual的第一个Pointwise卷积（Conv2d 1x1 卷积层） Quantize the first Pointwise convolution (Conv2d 1x1) in each InvertedResidual block
def quantize_inverted_residual_first_pointwise(model):
    for module in model.modules():
        if isinstance(module, InvertedResidual):
            # 量化第一个 Pointwise 卷积层 Quantize the first Pointwise convolution layer
            if isinstance(module.conv[0], nn.Conv2d):
                quantized_weight = torch.quantize_per_tensor(module.conv[0].weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                module.conv[0].weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_inverted_residual_first_pointwise(model)  # 量化每个InvertedResidual的第一个Pointwise卷积层 Quantize the first Pointwise convolution layer in each InvertedResidual
'''
'''
# 量化第一个InvertedResidual的第一个Pointwise卷积（Conv2d 1x1 卷积层） Quantize the first Pointwise convolution (Conv2d 1x1) in the first InvertedResidual block
def quantize_first_inverted_residual_first_pointwise(model):
    inverted_residual_found = False  # 标记是否已找到第一个InvertedResidual模块 Flag to check if the first InvertedResidual block is found
    for module in model.modules():
        if isinstance(module, InvertedResidual) and not inverted_residual_found:
            # 量化第一个 Pointwise 卷积层 Quantize the first Pointwise convolution layer
            if isinstance(module.conv[0], nn.Conv2d):
                quantized_weight = torch.quantize_per_tensor(module.conv[0].weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                module.conv[0].weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
            inverted_residual_found = True  # 设置标记表示已找到第一个InvertedResidual模块 Set the flag to true indicating the first InvertedResidual block is found
    return model

model = quantize_first_inverted_residual_first_pointwise(model)  # 量化第一个InvertedResidual的第一个Pointwise卷积层 Quantize the first Pointwise convolution layer in the first InvertedResidual
'''
'''
# 量化每一个InvertedResidual的最后一个Pointwise卷积（Conv2d 1x1 卷积层） Quantize the last Pointwise convolution (Conv2d 1x1) in each InvertedResidual block
def quantize_inverted_residual_last_pointwise(model):
    for module in model.modules():
        if isinstance(module, InvertedResidual):
            # 遍历InvertedResidual模块中的子层 Iterate over submodules in InvertedResidual
            for submodule in module.modules():
                # 如果子层是Conv2d并且卷积核为1x1 If submodule is Conv2d and kernel size is 1x1
                if isinstance(submodule, nn.Conv2d) and submodule.kernel_size == (1, 1):
                    quantized_weight = torch.quantize_per_tensor(submodule.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                    submodule.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_inverted_residual_last_pointwise(model)  # 量化每个InvertedResidual的最后一个Pointwise卷积层 Quantize the last Pointwise convolution layer in each InvertedResidual
'''
'''
# 量化第一个InvertedResidual的最后一个Pointwise卷积（Conv2d 1x1 卷积层） Quantize the last Pointwise convolution (Conv2d 1x1) in the first InvertedResidual block
def quantize_first_inverted_residual_last_pointwise(model):
    inverted_residual_found = False  # 标记是否已找到第一个InvertedResidual模块 Flag to check if the first InvertedResidual block is found
    for module in model.modules():
        if isinstance(module, InvertedResidual) and not inverted_residual_found:
            # 遍历InvertedResidual模块中的子层 Iterate over submodules in InvertedResidual
            for submodule in module.modules():
                # 如果子层是Conv2d并且卷积核为1x1 If submodule is Conv2d and kernel size is 1x1
                if isinstance(submodule, nn.Conv2d) and submodule.kernel_size == (1, 1):
                    quantized_weight = torch.quantize_per_tensor(submodule.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                    submodule.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
            inverted_residual_found = True  # 设置标记表示已找到第一个InvertedResidual模块 Set the flag to true indicating the first InvertedResidual block is found
            break
    return model

model = quantize_first_inverted_residual_last_pointwise(model)  # 量化第一个InvertedResidual的最后一个Pointwise卷积层 Quantize the last Pointwise convolution layer in the first InvertedResidual
'''
'''
# 将模型的分类器部分和最后一个 InvertedResidual 的权重量化为 int8 Quantize the weights of the model's classifier and the last InvertedResidual to int8
def quantize_model_tail(model):
    classifier_layers = [model.classifier]
    for layer in classifier_layers:
        if isinstance(layer, nn.Conv2d):
            quantized_weight = torch.quantize_per_tensor(layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
            layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    last_inverted_residual = model.features[-1]
    if isinstance(last_inverted_residual, nn.Sequential):
        for sub_layer in last_inverted_residual:
            if isinstance(sub_layer, nn.Conv2d):
                quantized_weight = torch.quantize_per_tensor(sub_layer.weight.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
                sub_layer.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device))
    return model

model = quantize_model_tail(model)  # 量化模型的分类器部分和最后一个 InvertedResidual Quantize the classifier part and the last InvertedResidual of the model
'''

# 计算模型参数量 Calculate model parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 打印模型参数量和大小等信息 Print model parameter count and size information
print("Model parameter count:", count_parameters(model))
print("Model size (MB):", sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2))  # Assume 32-bit float, so each parameter takes 4 bytes

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
