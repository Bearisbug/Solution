# 解决方法

## Python ![Pycharm](https://img.shields.io/badge/Pycharm-97D587?style=flat&logo=pycharm&logoColor=black&labelColor=dad7cd)

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

---

## Swift ![Static Badge](https://img.shields.io/badge/Xcode-7F3AD1?style=flat&logo=xcode&logoColor=black&labelColor=147EFB)
| 作用         | 快捷键                             |
| ------------- | --------------------------------- |
| 开启预览框      | `⌘` + `⇧` + `↩︎`                      |
| 继续或暂停      | `⌘` + `⌥` + `P`                      |
| 快速重命名      | `⌘` + `⌃` + `E`                      |
| 格式化代码      | `⌃` + `⇧` + `I`                      |

---
## TypeScript ![Static Badge](https://img.shields.io/badge/VSCode-3178C6?style=flat&logo=typescript&logoColor=3178C6&labelColor=white)
##### 1.经典布局形式：
```typescript
// 三行网格居中布局
<div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16">
  <div className="flex flex-row gap-8 row-start-2 items-center w-full h-full">
  </div>
</div>
```
---
## 常用网站

#### 徽章制作 ```https://shields.io/badges```

#### Logo资源 `https://simpleicons.org/`
