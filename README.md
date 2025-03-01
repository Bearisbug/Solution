# 解决方法

## <img src="https://img.icons8.com/?size=100&id=13441&format=png&color=000000" width="15"> Python

![Static Badge](https://img.shields.io/badge/PyCharm-white?style=social&logo=pycharm) ![Static Badge](https://img.shields.io/badge/PyTorch-white?style=social&logo=pytorch)

`checkpoint`

###### 1. PyTorch 模型权重文件

PyTorch 模型权重文件常见两种保存方式：

* **保存 state\_dict 的 `.pth` 文件**
  仅包含模型各层参数的 Python 字典，适合在训练过程中断点恢复或加载权重。
* **保存整个模型的 `.pt` 文件**
  包含模型结构与参数，便于部署和推理，但加载时通常需要对应的模型定义代码。

**查看权重结构示例代码：**

```python
import torch

# 加载保存的权重文件（假设保存的是 state_dict）
state_dict = torch.load('model_weights.pth', map_location='cpu')

print("模型权重文件中的键及对应张量形状：")
for key, tensor in state_dict.items():
    print(f"{key}: {tensor.shape}")
```

###### 2. Keras (HDF5 格式) 模型权重文件

Keras 常用 HDF5 文件格式（.h5）来保存模型结构、权重和训练配置。
使用 `h5py` 库可以查看文件内部的层级结构和属性信息。

```python
import h5py

def print_h5_structure(name, obj):
    print(name)
    for attr, value in obj.attrs.items():
        print(f"    {attr}: {value}")

with h5py.File('model_weights.h5', 'r') as f:
    print("HDF5 文件结构：")
    f.visititems(print_h5_structure)
```

###### 3. TensorFlow Checkpoint 模型权重文件

TensorFlow 的 checkpoint 文件通常包含多个文件，用于保存模型变量的数值。
使用 `tf.train.list_variables` 可以列出 checkpoint 中保存的所有变量及其形状。

```python
import tensorflow as tf

checkpoint_path = 'model.ckpt'
variables = tf.train.list_variables(checkpoint_path)

print("Checkpoint 文件中的变量名称及形状：")
for name, shape in variables:
    print(f"{name}: {shape}")
```

###### 4. ONNX 模型权重文件

ONNX 文件（.onnx）存储整个计算图及其参数信息，方便在不同深度学习框架间进行模型转换。
使用 `onnx` 库可以加载并查看 ONNX 模型的各部分信息。

```python
import onnx

model = onnx.load("model.onnx")
print("Graph inputs:")
for inp in model.graph.input:
    print(inp.name)
print("Graph outputs:")
for out in model.graph.output:
    print(out.name)
print("Graph initializers (parameters):")
for initializer in model.graph.initializer:
    print(initializer.name, initializer.dims)
```

###### 5. Caffe (.caffemodel) 模型权重文件

Caffe 的 `.caffemodel` 文件是二进制的 protobuf 文件，通常与对应的 `.prototxt` 模型定义文件配合使用。
使用 Caffe 的 Python 接口可以加载模型并查看各层参数的形状。

```python
import caffe

# 加载模型定义和权重文件
net = caffe.Net('deploy.prototxt', 'model.caffemodel', caffe.TEST)

print("Caffe 模型各层参数的形状：")
for layer_name, params in net.params.items():
    shapes = [p.data.shape for p in params]
    print(f"{layer_name}: {shapes}")
```

###### 6. Darknet / YOLO (.weights) 模型权重文件

Darknet 的 `.weights` 文件格式为二进制文件，其结构通常包括一个文件头和后续的权重数据。
文件头一般包含 5 个 int32 数值（如版本信息、训练样本数等），其余部分为 float32 类型的权重数据。

```python
import numpy as np

def read_darknet_weights(file_path):
    with open(file_path, "rb") as f:
        # Darknet 文件头：前 5 个 int32 数值
        header = np.fromfile(f, dtype=np.int32, count=5)
        print("Header:", header)
        # 剩余部分为权重，float32 格式
        weights = np.fromfile(f, dtype=np.float32)
        print("Total weights count:", len(weights))

read_darknet_weights("yolov3.weights")
```

`matplotlib`

##### 1.中文字体问题

在使用 `matplotlib` 绘制图表时，可能会遇到中文字体显示为方块或者无法显示负号的问题。这是因为 `matplotlib` 默认字体可能不支持中文。
**解决方法（设置指定支持中文的字体）：**

```python
import matplotlib.font_manager as fm

# 列出所有字体文件路径
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    print(font)
    
# 列出所有字体名称
print(sorted([f.name for f in fm.fontManager.ttflist]))

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # 设置找寻到的中文字体为
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
```

## <img src="https://img.icons8.com/?size=100&id=24465&format=png&color=000000" width="15"> Swift

![Static Badge](https://img.shields.io/badge/Xcode-white?style=social&logo=Xcode)

| 作用         | 快捷键                             |
| ------------- | --------------------------------- |
| 开启预览框      | `⌘` + `⇧` + `↩︎`                      |
| 继续或暂停      | `⌘` + `⌥` + `P`                      |
| 快速重命名      | `⌘` + `⌃` + `E`                      |
| 格式化代码      | `⌃` + `⇧` + `I`                      |

## <img src="https://img.icons8.com/?size=100&id=uJM6fQYqDaZK&format=png&color=000000" width="15"> TypeScript

![Static Badge](https://img.shields.io/badge/VSCodium-white?style=social&logo=vscodium)


##### 1.经典布局形式：

```typescript
// 三行网格居中布局
<div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16">
  <div className="flex flex-row gap-8 row-start-2 items-center w-full h-full">
  </div>
</div>
```

## <img src="https://img.icons8.com/?size=100&id=7AFcZ2zirX6Y&format=png&color=000000" width="15"> Dart

![Static Badge](https://img.shields.io/badge/Intellij_IDEA-white?style=social&logo=intellijidea) ![Static Badge](https://img.shields.io/badge/Flutter-white?style=social&logo=flutter)

## 常用网站

#### 徽章制作 `https://shields.io/badges`

#### Logo资源 `https://simpleicons.org/`

#### TechIcons `https://techicons.dev/`
