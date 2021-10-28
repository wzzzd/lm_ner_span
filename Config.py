
import os
import random




class Config(object):
    
    def __init__(self):
        
        # 运行模式
        self.mode = 'train'
        
        # GPU配置
        self.cuda_visible_devices = '0'                           # 可见的GPU
        self.device = 'cuda:0'                                      # master GPU
        self.port = str(random.randint(10000,60000))                # 多卡训练进程间通讯端口
        self.init_method = 'tcp://localhost:' + self.port           # 多卡训练的通讯地址
        self.world_size = 1                                         # 线程数，默认为1
        
        # 训练配置
        self.num_epochs = 30                                        # 迭代次数
        self.batch_size = 128                                     # 每个批次的大小
        self.learning_rate = 5e-5                                   # 学习率
        self.num_warmup_steps = 0.1                                 # warm up步数
        self.sen_max_length = 128                                   # 句子最长长度
        self.padding = True                                         # 是否对输入进行padding
        self.step_save = 100

        # 模型及路径配置
        self.initial_pretrain_model = 'bert-base-chinese'           # 加载的预训练分词器checkpoint，默认为英文。若要选择中文，替换成 bert-base-chinese
        self.initial_pretrain_tokenizer = 'bert-base-chinese'       # 加载的预训练模型checkpoint，默认为英文。若要选择中文，替换成 bert-base-chinese
        self.path_model_save = './checkpoint/'                      # 模型保存路径
        self.path_datasets = './data/CNER/'             # 数据集
        self.path_log = './logs/'
        self.path_output = './outputs/'
