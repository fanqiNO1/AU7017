# AU7017 智能信息处理课程大作业

本仓库为 SJTU AU7017 智能信息处理课程大作业仓库。

## 主要内容

### ViT

参考 LLM 的架构实现了 ViT。

训练方法：

```bash
python vit/main.py
```

### CPO

参考 DPO（直接偏好优化） 实现了 CPO（分类偏好优化）。

训练方法：

```bash
python cpo/main.py
```

### MnistVL

参考 LLaVA 架构，实现了用于 MNIST 数据集的多模态模型。

> \[!IMPORTANT\]
> MnistVL 部分暂无实验结果。

测试 `Qwen2.5-VL-3B-Instruct` 方法：

```bash
python mnistvl/test_qwenvl.py
```

相关测试报告将会输出于 `mnistvl` 目录。

### utils

`utils` 文件夹下主要是报告时相关脚本。

如数据集可视化、混淆矩阵可视化等。
