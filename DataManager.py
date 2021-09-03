
import re
import os
import random
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from utils.IOOption import open_file, write_text



class DataManager(object):
    
    def __init__(self, config):
        
        self.config = config
        # # 标签
        # self.label2ids = {x:i for i,x in enumerate(config.tag_type)}
        # self.ids2label = {i:x for i,x in enumerate(config.tag_type)}
        # 读取tokenizer分词模型
        self.tokenizer = AutoTokenizer.from_pretrained(config.initial_pretrain_tokenizer)    
    
    
    def get_dataset(self, data_type='train'):
        """获取数据集"""
        file = '{}.txt'.format(data_type)
        dataloader = self.data_process(file)
        return dataloader


    def data_process(self, file_name):
        """
        数据转换
        """
        # 获取数据
        src, tgt = open_file(self.config.path_datasets + file_name, sep='\t')
        src = [str(x) for x in src]
        # tgt = [str(x) for x in tgt]
        # 获取标签
        if self.config.mode=='train':
            tag = list(set([str(x).replace('B-','').replace('I-','') for line in tgt for x in line]))
            # tag = list(set([re.sub(r'[B-|I-]', '', str(x)) for line in tgt for x in line]))
            self.label2ids = {x:i for i,x in enumerate(tag)}
            self.ids2label = {i:x for i,x in enumerate(tag)}
            write_text(list(self.label2ids.keys()), self.config.path_datasets+'label.txt')
        else:
            tag = [ x.strip() for x in open(self.config.path_datasets+'label.txt','r').readlines()]
            self.label2ids = {x:i for i,x in enumerate(tag)}
            self.ids2label = {i:x for i,x in enumerate(tag)}
        dataset = pd.DataFrame({'src':src, 'labels':tgt})
        # dataset.to_csv('./data/cache.csv', sep='\t', index=False)
        # dataframe to datasets
        raw_datasets = Dataset.from_pandas(dataset)
        # tokenizer.
        tokenized_datasets = raw_datasets.map(lambda x: self.tokenize_function(x), batched=True)        # 对于样本中每条数据进行数据转换
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)                               # 对数据进行padding
        # label_entity = tokenized_datasets['label_entity']
        # label_entity = [ x for x in label_entity if x != []]
        tokenized_datasets = tokenized_datasets.remove_columns(["src","labels"])                        # 移除不需要的字段
        tokenized_datasets.set_format("torch")                                                           # 格式转换
        # 转换成DataLoader类
        # train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=config.batch_size, collate_fn=data_collator)
        # eval_dataloader = DataLoader(tokenized_datasets_test, batch_size=config.batch_size, collate_fn=data_collator)
        sampler = RandomSampler(tokenized_datasets) if not torch.cuda.device_count() > 1 else DistributedSampler(tokenized_datasets)
        dataloader = DataLoader(tokenized_datasets, sampler=sampler, batch_size=self.config.batch_size, collate_fn=data_collator)

        return dataloader


    def tokenize_function(self, example):
        """
        数据转换
        """
        # 分词
        # src = [self.tokenizer.convert_tokens_to_string(x.split(' ')) for x in example["src"]]
        # token = self.tokenizer(src, truncation=True, max_length=self.config.sen_max_length, padding=self.config.padding)
        token = self.tokenizer(example["src"], truncation=True, max_length=self.config.sen_max_length, padding=self.config.padding)
        # 标签处理
        # 获取标签实体
        # label_entity = [ eval(line) for line in example['labels']]
        label_entity = [self.get_entity(line[:self.config.sen_max_length-2]) for line in example['labels']]
        # lab = [ x for x in label_entity if x != []]
        # 获取表示首尾label向量
        label_start = []
        label_end = []
        for line in label_entity:
            start_ids = [0] * self.config.sen_max_length
            end_ids = [0] * self.config.sen_max_length
            for entity in line:
                # 样本的开始和结果索引,因为token第一位加入了[CLS]，所以index要+1
                tmp_label = entity[0]
                tmp_start = entity[1] + 1
                tmp_end = entity[2] + 1
                # 赋值给两个ids向量
                tmp_label_ids = self.label2ids[tmp_label]                
                start_ids[tmp_start] = tmp_label_ids
                end_ids[tmp_end] = tmp_label_ids
            label_start.append(start_ids)
            label_end.append(end_ids)
        # 添加标签到样本中
        token.data['label_start'] = label_start
        token.data['label_end'] = label_end
        # token.data['label_entity'] = label_entity
        return token


    def get_entity(self, seq):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
                chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx

                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks



