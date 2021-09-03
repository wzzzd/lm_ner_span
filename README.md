# LM_SPAN
基于实体首尾指针SPAN的 序列标注/命名实体识别 框架
主要框架是pytorch，base model使用的是bert，基于huggingface/transformers。


## 思路
将序列标注任务，转换成预测实体在文本中的实体头部的索引和实体尾部的索引


## Requirement
```
    python3.6
    numpy==1.19.5
    pandas==1.1.3
    torch==1.3.0
    transformers==4.6.1
```
可通过以下命令安装依赖包
```
    pip install -r requirement.txt
```

