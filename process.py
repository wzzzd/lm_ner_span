
import os
from posixpath import sep
import time
import random
import logging
import math
import numpy as np
import pandas as pd
import torch

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




def train(config, train_ld, valid_loader, test_loader):
    """
        预训练模型
    """
    # train_ld = train_loader[0]
    # train_label = train_loader[1]
    
    print('training start')
    # 初始化配置
    device = torch.device(config.device)

    # 初始化模型和优化器
    print('model loading')
    tag = [ x.strip() for x in open(config.path_datasets+'label.txt','r').readlines()]
    num_labels = len(tag)
    model_config = BertConfig.from_pretrained(config.initial_pretrain_model, num_labels=num_labels)
    model = BertSpanForNer.from_pretrained(config.initial_pretrain_model, config=model_config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # 定义优化器配置
    num_training_steps = config.num_epochs * len(train_ld)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 分布式训练
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                    find_unused_parameters=True,
                                                    broadcast_buffers=True)
    print('start to train')
    model.train()
    step_total = config.num_epochs * len(train_ld)
    step_current = 0
    f1_best = 0
    for epoch in range(config.num_epochs):
        # print('Training Epoch: {0}'.format(epoch))
        progress_bar = ProgressBar(n_total=len(train_ld), desc='Training epoch:{0}'.format(epoch))
        # progress_bar = tqdm(range(len(train_ld)))
        for i, batch in enumerate(train_ld):
            batch.data = {k:v.to(device) for k,v in batch.data.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)
            progress_bar(i, {'loss': loss.item()})
            # if step_current % 500 == 0:
            #     print('Training epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, step_current, step_total, loss.item()))

            if step_current%config.step_save==0 and step_current>0:
                # 模型评估
                f1_eval = eval(valid_loader, model, tag, config)
                # 模型保存
                if f1_eval != 0:
                    path = config.path_model_save + 'step_{}/'.format(step_current)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    model_save = model.module if torch.cuda.device_count() > 1 else model
                    model_save.save_pretrained(path)
                    print('saving model: {}'.format(path))
                    if f1_eval > f1_best:
                        path = config.path_model_save + 'step_best/'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        model_save = model.module if torch.cuda.device_count() > 1 else model
                        model_save.save_pretrained(path)
                        print('saving best model: {}'.format(path))
                        
            step_current += 1
        print('\nIter:{0}/{1}  loss:{2}\n'.format(step_current, step_total, loss.item()))


def eval(valid_ld, model, tag, config):
    # 拆分数据
    # valid_ld = valid_loader[0]
    # valid_label = valid_loader[1]
    # 定义metric
    id2label = {i:x for i, x in enumerate(tag)}
    metric = SpanEntityScore(id2label)
    losses = []
    model.eval()
    for i, batch in enumerate(valid_ld):
        # 推断
        batch.data = {k:v.to(config.device) for k,v in batch.data.items()}
        with torch.no_grad():
            outputs = model(**batch)
        tmp_eval_loss, start_logits, end_logits = outputs[:3]
        # 计算损失
        losses.append(tmp_eval_loss)
        # 计算指标
        # label = valid_label[i]
        # 计算指标
        # 区分是ground true还是prediction
        start_lab = batch.data['label_start'].cpu().numpy()[:,1:-1]
        end_lab = batch.data['label_end'].cpu().numpy()[:,1:-1]
        start_pred = torch.argmax(start_logits, -1)[:,1:-1].cpu().numpy()
        end_pred = torch.argmax(end_logits, -1)[:,1:-1].cpu().numpy()
        for i in range(len(start_lab)):
            label = bert_extract_item(start_lab[i], end_lab[i])
            pred = bert_extract_item(start_pred[i], end_pred[i])
            metric.update(true_subject=label, pred_subject=pred)
        # label = bert_extract_item(batch.data['label_start'], batch.data['label_end'])
        # pred = bert_extract_item(start_logits, end_logits)
        # metric.update(true_subject=label, pred_subject=pred)
    eval_info, entity_info = metric.result()
    print('\nEval  precision:{0}  recall:{1}  f1:{2}'.format(round(eval_info['acc'],4), round(eval_info['recall'],4), round(eval_info['f1'],4)))
    for item in entity_info.keys():
        print('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}'.format(item, round(entity_info[item]['acc'],4), round(entity_info[item]['recall'],4), round(entity_info[item]['f1'],4)))
    return eval_info['f1']
    

def predict(config, test_loader):
    """
    预测
    """
    print('predict start')
    # 初始化配置
    device = torch.device(config.device)
    # 初始化模型和优化器
    print('model loading')
    path_model = config.path_model_save + 'step_best/'
    if not os.path.exists(path_model):
        print('model checkpoint file not exist!')
        return 
    
    tokenizer = BertTokenizer.from_pretrained(config.initial_pretrain_model)
    model = BertSpanForNer.from_pretrained(path_model)
    model.to(device)
    model.eval()
    # 初始化指标计算
    tag = [ x.strip() for x in open(config.path_datasets+'label.txt','r').readlines()]
    id2label = {i:x for i, x in enumerate(tag)}
    metric = SpanEntityScore(id2label)
    
    progress_bar = ProgressBar(n_total=len(test_loader), desc='Predict')
    src = []
    tgt_label = []
    tgt_pred = []
    for i, batch in enumerate(test_loader):
        # 推断
        batch.data = {k:v.to(config.device) for k,v in batch.data.items()}
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
            tmp_label = bert_extract_item(start_lab[i], end_lab[i])
            tmp_pred = bert_extract_item(start_pred[i], end_pred[i])
            metric.update(true_subject=tmp_label, pred_subject=tmp_pred)
            label.append(tmp_label)
            pred.append(tmp_pred)
        # label = bert_extract_item(batch.data['label_start'], batch.data['label_end'])
        # pred = bert_extract_item(start_logits, end_logits)
        # metric.update(true_subject=label, pred_subject=pred)
        # 文本抽取转化
        mask =  [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
        tmp_src = [[x for x in tokenizer.convert_ids_to_tokens(line) if x not in mask] for line in batch.data['input_ids']]
        tmp_label = [token_extract_tag(s, l, id2label) for s, l in zip(tmp_src, label)]
        tmp_pred = [token_extract_tag(s, p, id2label) for s, p in zip(tmp_src, pred)]
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
    if not os.path.exists(config.path_output):
        os.mkdir(config.path_output)
    data.to_csv(config.path_output+'pred_data.csv', sep='\t', index=False)
    
    return eval_info['f1']
    

def token_extract_tag(src, tgt, id2label):
    tag = {}
    for line in tgt:
        label = id2label[line[0]]
        start = line[1]
        end = line[2]
        tmp_tag = src[start:end+1]
        tag.setdefault(label, [])
        tag[label].append(tmp_tag)
    return tag
    
    
    



if __name__ == '__main__':
    
    config = Config()
    train(config)
    # load_lm()
