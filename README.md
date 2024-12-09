# 基于 XLM-RoBERTa 和 Seq2Seq 框架的多语言自然语言推理（NLI）模型

本项目实现了一种用于多语言自然语言推理（NLI）任务的混合方法，解决了 **语义多样性** 和 **低资源语言数据稀缺** 等挑战。模型结合了 **XLM-RoBERTa**、Seq2Seq 架构、**对比学习（SimCSE）** 和数学优化技术，显著提升了多语言推理任务的性能。

## 目录
- [项目概述](#项目概述)
- [功能特点](#功能特点)
- [安装方法](#安装方法)
- [使用说明](#使用说明)
- [实验结果](#实验结果)
- [许可证](#许可证)

## 项目概述
自然语言推理（NLI）是自然语言处理（NLP）中的一项核心任务，主要目标是判断句子对之间的语义关系，如：
- **蕴涵（Entailment）**
- **矛盾（Contradiction）**
- **无关（Neutral）**

本项目针对多语言 NLI 的挑战，提出了以下解决方案：
1. 利用 **XLM-RoBERTa** 进行多语言预训练。
2. 采用 **Seq2Seq 框架** 建模句子间的依赖关系。
3. 使用数据增强与聚类技术提升低资源语言的性能。

实验基于来自 Kaggle 的 **多语言数据集**（覆盖 15 种语言）进行评估。

## 功能特点
- 支持 **15 种语言** 的多语言自然语言推理。
- 针对低资源语言的增强技术：
  - **对比学习（SimCSE）** 优化语义对齐。
  - **贝叶斯优化** 进行超参数调优。
  - **数据增强** 和 **聚类算法** 平衡数据集分布。
- 提供从训练到评估的完整流程。

## 安装方法
### 1. 克隆代码库：
```bash
   git clone https://github.com/Jeffjeno/Artificial-Intelligence-Course-Design.git
   cd Artificial-Intelligence-Course-Design
```

### 2.	安装依赖库：
```bash
pip install -r requirements.txt
```

**使用说明**
#### 1.	准备数据集：
- 从 Kaggle 下载多语言数据集。
- 将数据集放置在 data/ 文件夹中。
#### 2.	训练模型：
```bash
  python train.py --config config.yaml
```

#### 3.	评估模型：
```bash
  python evaluate.py --checkpoint checkpoints/model.pth
```
#### 4.	使用贝叶斯优化调整超参数：
```bash
  python optimize.py 
```
## 实验结果

-	高资源语言 的准确率超过 90%。
-	低资源语言 的准确率从 60% 提升至 80%。

# 贡献

欢迎贡献代码！请通过提交 PR 或创建 issue 的方式参与项目。

# 许可证

本项目基于 MIT 许可证进行发布，详情请参阅 LICENSE 文件。

# 致谢

-	XLM-RoBERTa: https://arxiv.org/abs/1911.02116
-	SimCSE: https://github.com/princeton-nlp/SimCSE
-	Seq2Seq 框架: 适用于生成任务的多种实现。

如需更多信息，请参考我们的论文。

### 注意事项
1. 请根据实际情况修改文件和脚本名称（如 `train.py` 或 `config.yaml`）。
2. 如果您还未上传 `requirements.txt` 或相关配置文件，请确认内容完整。
3. 确保链接和引用的文件路径正确有效。
