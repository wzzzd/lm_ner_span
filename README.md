# LM_SPAN
基于实体首尾指针SPAN的 序列标注/命名实体识别 框架。

能解决嵌套实体识别问题。

基于pytorch，base model使用的是bert，基于huggingface/transformers。


## 思路
将序列标注任务，转换成预测实体在文本中的实体头部的索引和实体尾部的索引

<img src=./file/pic.jpg width=100% />


## 数据集
* **CNER**
    * 关于简历的的数据。
    * 数据分为8类标签类别，分别为：
    ```
        国家（CONT）
        民族（RACE）
        姓名（NAME）
        组织（ORG）
        地址（LOC）
        专业（PRO）
        学历（EDU）
        职称（TITLE）
    ```


## 训练过程
<img src=./file/train.png width=100% />


## 支持的训练模式

- 混合精度训练
- GPU多卡训练
- 对抗训练：FGM/PGD
- 支持中英文语料训练


## Requirement
```
    python3.6
    numpy==1.19.5
    pandas==1.1.3
    torch==1.3.0
    datasets==1.10.2
    transformers==4.6.1
```
可通过以下命令安装依赖包
```
    pip install -r requirement.txt
```


## Get Start
运行以下命令
```
    python run.py
```
