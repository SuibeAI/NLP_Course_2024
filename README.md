# NLP_Course_2024
克隆代码库
```bash
git clone https://github.com/SuibeAI/NLP_Course_2024.git
```
解压GloVe数据
```bash
cd NLP_Course_2024/assignments/assignment1_glove 
unzip glove.6B.50d.txt.zip 
cd ../..
```

## 作业1：GloVe向量
1. 完成find_nearest_word函数，实现根据当前词查找最近词（不包含自身单词）
2. 完成find_analogy函数，实现类别关系a之于b，相当于c之于(不包含单词a,b,c) 
3. 使用Numpy向量化（vectorized）方式完成find_nearest_word_vectorizedversion函数，提升效率。


## 作业2：文本情感分类
1. 跑通模型
2. 尝试修改代码，如改变模型的input_dim, hid_dim的值，改变Vocabulary的值，甚至改变模型层数或类型，观察指标变化



