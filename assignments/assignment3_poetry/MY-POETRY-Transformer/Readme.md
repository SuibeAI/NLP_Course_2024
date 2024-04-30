# 神经网络写诗
## 数据
1. 下载中国诗歌数据集[https://github.com/chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

2. 参照dataset.py中的操作，从chinese-poetry/全唐诗/poet.song.*.json中提取所有五言绝句，用于模型训练。

## 模型
1. 使用基于Transformer的模型，如GPT，BERT，BART等进行设计，不可以直接调用Pytorch的Transformer模型库，需要自己编写模型（类似课堂上的attention is all you need注解）。

## 目标
完成模型训练后，模型需要具备以下推理能力。
1. 模型能够根据标题，生成五言绝句诗
   ```python
   content = generate_poetry(title="夏日繁花")
   print(content)
   ```
2. 模型能够根据标题和前小部分诗歌，生成完整的五言绝句诗
   ```python
   content = generate_poetry(title="夏日繁花", content="夏日校园中,")
   print(content)
   ```

## 完善代码
检查所有Todo位置

## 训练模型
训练
```
cd assignments/assignment3_poetry/MY-POETRY-Transformer
python train.py
```

## 测试模型
测试
```
cd assignments/assignment3_poetry/MY-POETRY-Transformer
python demo.py
```