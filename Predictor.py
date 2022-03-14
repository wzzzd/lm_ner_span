
import os
from posixpath import sep
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch
from apex import amp
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig
# from model.BertForMaskedLM import BertForMaskedLM
from model.BertSpan import BertSpanForNer
from Config import Config
from model.metrics.ner_metrics import SpanEntityScore, bert_extract_item
from utils.progressbar import ProgressBar
from model.optimal.adversarial import FGM,PGD



class Predictor(object):
    
    def __init__(self, config, test_loader):
        self.config = config
        self.test_loader = test_loader
    
    
    def predict(self):
        """
        预测
        """
        print('predict start')
        # 初始化配置
        device = torch.device(self.config.device)
        # 初始化模型和优化器
        print('model loading')
        path_model = self.config.path_model_save + 'step_best/'
        if not os.path.exists(path_model):
            print('model checkpoint file not exist!')
            return 

        tokenizer = BertTokenizer.from_pretrained(self.config.initial_pretrain_model)
        model = BertSpanForNer.from_pretrained(path_model)
        model.to(device)
        model.eval()

        # 混合精度
        if self.config.fp16:
            model = amp.initialize(model, opt_level='O3')

        # 初始化指标计算
        tag = [ x.strip() for x in open(self.config.path_datasets+'label.txt','r').readlines()]
        id2label = {i:x for i, x in enumerate(tag)}
        label2id = {x:i for i, x in enumerate(tag)}
        metric = SpanEntityScore(id2label)

        progress_bar = ProgressBar(n_total=len(self.test_loader), desc='Predict')
        src = []
        tgt_label = []
        tgt_pred = []
        for i, batch in enumerate(self.test_loader):
            # 推断
            batch.data = {k:v.to(self.config.device) for k,v in batch.data.items()}
            with torch.no_grad():
                outputs = model(**batch)
            _, start_logits, end_logits = outputs[:3]
            # 计算指标
            # 区分是ground true还是prediction
            start_lab = batch.data['label_start'].cpu().numpy()[:,1:-1]
            end_lab = batch.data['label_end'].cpu().numpy()[:,1:-1]
            start_pred = torch.argmax(start_logits, -1)[:,1:-1].cpu().numpy()
            end_pred = torch.argmax(end_logits, -1)[:,1:-1].cpu().numpy()
            label = []
            pred = []
            for i in range(len(start_lab)):
                tmp_label = bert_extract_item(start_lab[i], end_lab[i], label2id)
                tmp_pred = bert_extract_item(start_pred[i], end_pred[i], label2id)
                metric.update(true_subject=tmp_label, pred_subject=tmp_pred)
                label.append(tmp_label)
                pred.append(tmp_pred)
            # label = bert_extract_item(batch.data['label_start'], batch.data['label_end'])
            # pred = bert_extract_item(start_logits, end_logits)
            # metric.update(true_subject=label, pred_subject=pred)
            # 文本抽取转化
            mask =  [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
            tmp_src = [[x for x in tokenizer.convert_ids_to_tokens(line) if x not in mask] for line in batch.data['input_ids']]
            tmp_label = [self.token_extract_tag(s, l, id2label) for s, l in zip(tmp_src, label)]
            tmp_pred = [self.token_extract_tag(s, p, id2label) for s, p in zip(tmp_src, pred)]
            tmp_src_string = [tokenizer.convert_tokens_to_string(x) for x in tmp_src]
            # tmp_label_string = [tokenizer.convert_tokens_to_string(x) for x in tmp_label]
            # tmp_pred_string = [tokenizer.convert_tokens_to_string(x) for x in tmp_pred]
            tmp_label_string = [{k:[tokenizer.convert_tokens_to_string(x) for x in v] for k,v in line.items()} for line in tmp_label]
            tmp_pred_string = [{k:[tokenizer.convert_tokens_to_string(x) for x in v] for k,v in line.items()} for line in tmp_pred]


            src.extend(tmp_src_string)
            tgt_label.extend(tmp_label_string)
            tgt_pred.extend(tmp_pred_string)
            progress_bar(i, {})

        eval_info, entity_info = metric.result()
        print('\nEval  precision:{0}  recall:{1}  f1:{2}'.format(eval_info['acc'], eval_info['recall'], eval_info['f1']))
        for item in entity_info.keys():
            print('-- item:{0}  precision:{1}  recall:{2}  f1:{3}'.format(item, entity_info[item]['acc'], entity_info[item]['recall'], entity_info[item]['f1']))
        # 保存
        data = {'src':src, 'tgt_label':tgt_label, 'tgt_pred':tgt_pred}
        data = pd.DataFrame(data)
        if not os.path.exists(self.config.path_output):
            os.mkdir(self.config.path_output)
        data.to_csv(self.config.path_output+'pred_data.csv', sep='\t', index=False)

        return eval_info['f1']




    def token_extract_tag(self, src, tgt, id2label):
        tag = {}
        for line in tgt:
            label = id2label[line[0]]
            start = line[1]
            end = line[2]
            tmp_tag = src[start:end+1]
            tag.setdefault(label, [])
            tag[label].append(tmp_tag)
        return tag
    
