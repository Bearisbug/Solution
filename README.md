# <img src="https://s1.locimg.com/2025/03/04/28cfb15e65029.png" width=20> 虫の方法

## <img src="https://img.icons8.com/?size=100&id=13441&format=png&color=000000" width="15"> Python

[![Static Badge](https://img.shields.io/badge/PyCharm-white?style=social&logo=pycharm)](https://www.jetbrains.com/pycharm/) [![Static Badge](https://img.shields.io/badge/PyTorch-white?style=social&logo=pytorch)](https://pytorch.org/)

`checkpoint`
##### 权重文件结构

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

##### 中文字体问题

###### 在使用 `matplotlib` 绘制图表时，可能会遇到中文字体显示为方块或者无法显示负号的问题。这是因为 `matplotlib` 默认字体可能不支持中文。


**解决方法（设置指定支持中文的字体）：**

```python
import matplotlib.font_manager as fm

# 列出所有字体文件路径
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    print(font)
    
# 列出所有字体名称
print(sorted([f.name for f in fm.fontManager.ttflist]))

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # 或者 plt.rcParams['font.family'] = 'PingFang HK'  
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
```

`openai`
###### 简易 API 调用示例
```python
from openai import OpenAI

client = OpenAI(api_key="<Your API Key>")
client.base_url = "<Your API URL>"

# 完整输出
response = client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "user", "content": "你是谁"}
    ]
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="deepseek-r1",
    messages=[{"role": "user", "content": "根据中国古代圣人孔子的思想，人生的意义是什么？"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```
`os`
##### 文件遍历
###### 使用 os 库可以方便地遍历指定目录及其子目录中的所有文件和文件夹。这在批量处理文件或者查找特定文件时非常有用。
```python
import os

def traverse_directory(path):
    # os.walk 会递归遍历目录，返回三元组 (当前目录, 子目录列表, 文件列表)
    # root 是当前遍历到的目录。
    # dirs 是当前目录下的子目录名列表（不包含路径）。
    # files 是当前目录下的所有文件名列表（不包含路径）。
    for root, dirs, files in os.walk(path):
        print("当前目录：", root)
        print("子目录：", dirs)
        print("文件：", files)
        print("-" * 40)

# 示例：遍历当前目录
traverse_directory(".")
```

##### 设置环境变量
###### 在 Python 程序中，有时需要设置或修改环境变量（常用于设置代理或者镜像源）来控制程序行为。可以使用 os.environ 这个字典来进行环境变量的读取和设置。
```python


import os

# 设置环境变量 ( HTTP 和 HTTPS 代理作为示例 )
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 获取环境变量
value = os.environ.get('HTTP_PROXY')
print("HTTP_PROXY:", value)

# 判断某个环境变量是否存在
if 'HTTP_PROXY' in os.environ:
    print("环境变量 HTTP_PROXY 已设置")
else:
    print("环境变量 HTTP_PROXY 未设置")
```

## <img src="https://img.icons8.com/?size=100&id=24465&format=png&color=000000" width="15"> Swift

[![Static Badge](https://img.shields.io/badge/Xcode-white?style=social&logo=Xcode)](https://developer.apple.com/xcode/)

| 作用       | 快捷键                       |
| ------------ | ------------------------------ |
| 开启预览框 | `⌘`+`⇧`+`↩︎` |
| 继续或暂停 | `⌘`+`⌥`+`P`    |
| 快速重命名 | `⌘`+`⌃`+`E`    |
| 格式化代码 | `⌃`+`⇧`+`I`    |

##### 导入项目之后无法预览 View
###### View 报错: Select a scheme that builds a target which contains the current file, or add this file to a target that is built by the current scheme. 尝试输入:
```bash
rm -rf ~/Library/Developer/Xcode/DerivedData
```
## <img src="https://img.icons8.com/?size=100&id=uJM6fQYqDaZK&format=png&color=000000" width="15"> TypeScript

[![Static Badge](https://img.shields.io/badge/VSCodium-white?style=social&logo=vscodium)](https://vscodium.com/) [![Static Badge](https://img.shields.io/badge/TailWindCSS-white?style=social&logo=tailwindcss)](https://tailwindcss.com/)

`fullPage.js`

###### fullPage.js 是一个用于创建全屏滚动网站的 JavaScript 库，适用于单页面应用（SPA）或需要流畅滚动效果的网页。它提供了一种简单的方法来创建全屏滚动、滑动、分屏滚动等效果。

> 在 NextJS 中的示例使用代码如下：

```typescript
"use client";

import React, { useEffect } from "react";
import dynamic from "next/dynamic";
import "fullpage.js/dist/fullpage.css";

// 动态导入 fullpage.js，确保它只在客户端加载
const FullPageComponent = () => {
  useEffect(() => {
    const loadFullPage = async () => {
      if (typeof window !== "undefined") {
        //@ts-ignore
        const fullpage = (await import("fullpage.js")).default;
        new fullpage("#fullpage", {
          // 配置选项
        });
      }
    };

    loadFullPage();
  }, []);

  return (
    <div id="fullpage">
      <div className="section">第一屏内容</div>
      <div className="section">第二屏内容</div>
      <div className="section">第三屏内容</div>
    </div>
  );
};

export default FullPageComponent;
```

`Flex & Grid`

#### **1️⃣ Flexbox 对应表**

| **功能**                         | **CSS 代码**                    | **Tailwind CSS 类名** |
| ---------------------------------------- | --------------------------------------- | ----------------------------- |
| **Flex 容器**                    | `display: flex;`                  | `flex`                  |
| **内联 Flex 容器**               | `display: inline-flex;`           | `inline-flex`           |
| **主轴方向**                     | `flex-direction: row;`            | `flex-row`              |
|                                        | `flex-direction: row-reverse;`    | `flex-row-reverse`      |
|                                        | `flex-direction: column;`         | `flex-col`              |
|                                        | `flex-direction: column-reverse;` | `flex-col-reverse`      |
| **主轴对齐（justify-content）**  | `justify-content: flex-start;`    | `justify-start`         |
|                                        | `justify-content: center;`        | `justify-center`        |
|                                        | `justify-content: flex-end;`      | `justify-end`           |
|                                        | `justify-content: space-between;` | `justify-between`       |
|                                        | `justify-content: space-around;`  | `justify-around`        |
|                                        | `justify-content: space-evenly;`  | `justify-evenly`        |
| **交叉轴对齐（align-items）**    | `align-items: flex-start;`        | `items-start`           |
|                                        | `align-items: center;`            | `items-center`          |
|                                        | `align-items: flex-end;`          | `items-end`             |
|                                        | `align-items: stretch;`           | `items-stretch`         |
| **多行对齐（align-content）**    | `align-content: flex-start;`      | `content-start`         |
|                                        | `align-content: center;`          | `content-center`        |
|                                        | `align-content: flex-end;`        | `content-end`           |
|                                        | `align-content: space-between;`   | `content-between`       |
|                                        | `align-content: space-around;`    | `content-around`        |
| **换行**                         | `flex-wrap: wrap;`                | `flex-wrap`             |
|                                        | `flex-wrap: nowrap;`              | `flex-nowrap`           |
|                                        | `flex-wrap: wrap-reverse;`        | `flex-wrap-reverse`     |
| **子元素顺序（order）**          | `order: 1;`                       | `order-1`               |
|                                        | `order: -1;`                      | `order-first`           |
|                                        | `order: 9999;`                    | `order-last`            |
| **子元素扩展（flex-grow）**      | `flex-grow: 1;`                   | `flex-grow`             |
|                                        | `flex-grow: 0;`                   | `flex-grow-0`           |
| **子元素收缩（flex-shrink）**    | `flex-shrink: 1;`                 | `flex-shrink`           |
|                                        | `flex-shrink: 0;`                 | `flex-shrink-0`         |
| **子元素基础大小（flex-basis）** | `flex-basis: 100px;`              | `basis-[100px]`         |
| **子元素 flex 缩写**             | `flex: 1 1 0;`                    | `flex-1`                |
|                                        | `flex: 0 0 auto;`                 | `flex-none`             |

---

#### **2️⃣ Grid 对应表**

| **功能**                            | **CSS 代码**                           | **Tailwind CSS 类名**  |
| ------------------------------------------- | ---------------------------------------------- | ------------------------------ |
| **Grid 容器**                       | `display: grid;`                         | `grid`                   |
| **内联 Grid 容器**                  | `display: inline-grid;`                  | `inline-grid`            |
| **列数（grid-template-columns）**   | `grid-template-columns: repeat(3, 1fr);` | `grid-cols-3`            |
|                                           | `grid-template-columns: 1fr 2fr;`        | `grid-cols-[1fr_2fr]`    |
| **行数（grid-template-rows）**      | `grid-template-rows: repeat(3, 100px);`  | `grid-rows-3`            |
|                                           | `grid-template-rows: 100px auto;`        | `grid-rows-[100px_auto]` |
| **列间距（column-gap）**            | `column-gap: 16px;`                      | `gap-x-4`                |
| **行间距（row-gap）**               | `row-gap: 16px;`                         | `gap-y-4`                |
| **网格间距（gap）**                 | `gap: 16px;`                             | `gap-4`                  |
| **子元素跨列（grid-column）**       | `grid-column: span 2;`                   | `col-span-2`             |
|                                           | `grid-column: span 3 / span 3;`          | `col-span-3`             |
| **子元素跨行（grid-row）**          | `grid-row: span 2;`                      | `row-span-2`             |
| **对齐方式（justify-items）**       | `justify-items: start;`                  | `justify-items-start`    |
|                                           | `justify-items: center;`                 | `justify-items-center`   |
|                                           | `justify-items: end;`                    | `justify-items-end`      |
|                                           | `justify-items: stretch;`                | `justify-items-stretch`  |
| **对齐方式（align-items）**         | `align-items: start;`                    | `items-start`            |
|                                           | `align-items: center;`                   | `items-center`           |
|                                           | `align-items: end;`                      | `items-end`              |
|                                           | `align-items: stretch;`                  | `items-stretch`          |
| **内容对齐方式（justify-content）** | `justify-content: start;`                | `justify-start`          |
|                                           | `justify-content: center;`               | `justify-center`         |
|                                           | `justify-content: end;`                  | `justify-end`            |
|                                           | `justify-content: space-between;`        | `justify-between`        |
|                                           | `justify-content: space-around;`         | `justify-around`         |
|                                           | `justify-content: space-evenly;`         | `justify-evenly`         |
| **内容对齐方式（align-content）**   | `align-content: start;`                  | `content-start`          |
|                                           | `align-content: center;`                 | `content-center`         |
|                                           | `align-content: end;`                    | `content-end`            |
|                                           | `align-content: space-between;`          | `content-between`        |
|                                           | `align-content: space-around;`           | `content-around`         |
| **子元素对齐方式（place-self）**    | `place-self: center;`                    | `place-self-center`      |
| **自动布局（auto-flow）**           | `grid-auto-flow: row;`                   | `grid-flow-row`          |
|                                           | `grid-auto-flow: column;`                | `grid-flow-col`          |

`tailwindcss`
##### TailWindCSS V4 UI 适配问题
```css
/* 将 app.css 的 @import "tailwindcss" 替换为以下内容 */
/* 可解决 MUI 等主流 UI 库的适配问题。*/
@layer theme, base, components, utilities;
@import "tailwindcss/theme.css" layer(theme);
- @import "tailwindcss/preflight.css" layer(base);
@import "tailwindcss/utilities.css" layer(utilities);
```
## <img src="https://img.icons8.com/?size=100&id=7AFcZ2zirX6Y&format=png&color=000000" width="15"> Dart

[![Static Badge](https://img.shields.io/badge/Intellij_IDEA-white?style=social&logo=intellijidea)](https://www.jetbrains.com/idea/) [![Static Badge](https://img.shields.io/badge/Flutter-white?style=social&logo=flutter)](https://flutter.dev/)
##### 第三方库汇总
###### UI 库： [Forui](https://forui.dev/docs) [FL-Chart](https://pub.dev/packages/fl_chart)
###### 工具库：[location](https://pub.dev/packages/location) [flutter_launcher_icons](https://pub.dev/packages/flutter_launcher_icons)
##### iOS 和 Android 环境快速配置
```bash
# iOS
# 确保已经安装 Xcode
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
brew install cocoapods
# 进入项目的 ios 文件夹
pod install

# Android
# 确保安装了 Android Studio 如果没有请输入
brew install android-studio
# 在 SDK Manage > SDK Tools 中安装 Android SDK Command-Line Tools
# 接受license
flutter doctor --android-licenses

# NDK 版本更新
# 找到/opt/homebrew/Caskroom/flutter/3.29.2/flutter/packages/flutter_tools/gradle/src/main/groovy/flutter.groovy
# 修改默认的 version 即可
```
## 🪜 & 🧙‍♀️
[![Static Badge](https://img.shields.io/badge/Key-white?style=social&logo=webauthn)](https://acl4ssr-sub.github.io/)
```bash
# X-UI
bash <(curl -Ls https://raw.githubusercontent.com/mhsanaei/3x-ui/master/install.sh)
iptables -I INPUT -p tcp --dport 443 -j ACCEPT
iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo ufw disable
```
## 常用网站

#### 徽章制作 `https://shields.io/badges`

#### Logo资源 `https://simpleicons.org/`

#### TechIcons `https://techicons.dev/`

