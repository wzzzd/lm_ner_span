
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




class Trainer(object):
    
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.config = config
        self.device = torch.device(self.config.device)
        # 加载数据集
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # 加载标签
        self.tag = [ x.strip() for x in open(config.path_datasets+'label.txt','r').readlines()]
        # 加载模型
        self.load_model()


    def load_model(self):
        """
        加载模型
        """
        print('model loading')
        model_config = BertConfig.from_pretrained(self.config.initial_pretrain_model, num_labels=len(self.tag))
        self.model = BertSpanForNer.from_pretrained(self.config.initial_pretrain_model, config=model_config)
        self.model.to(self.device)
        print('>>>>>>>> mdoel structure >>>>>>>>')
        for name,parameters in self.model.named_parameters():
            print(name,':',parameters.size())
        print('>>>>>>>> mdoel structure >>>>>>>>')
        

    def train(self):
        """
            预训练模型
        """
        # weight decay
        bert_parameters = self.model.bert.named_parameters()
        start_parameters = self.model.start_fc.named_parameters()
        end_parameters = self.model.end_fc.named_parameters()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01, 'lr': self.config.learning_rate},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config.learning_rate},
            {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01, 'lr': 0.001},
            {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': 0.001},
            {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01, 'lr': 0.001},
            {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': 0.001}]
        step_total = self.config.num_epochs * len(self.train_loader) * self.config.batch_size
        # step_total = 640 #len(train_ld)*config.batch_size // config.num_epochs
        warmup_steps = int(step_total * self.config.num_warmup_steps)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=1e-8)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=step_total)
        # # 定义优化器配置
        # num_training_steps = config.num_epochs * len(train_ld)
        # optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        # lr_scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=config.num_warmup_steps,
        #     num_training_steps=num_training_steps
        # )

        # 分布式训练
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                        find_unused_parameters=True,
                                                        broadcast_buffers=True)
        # 对抗训练
        if self.config.adv_option == 'FGM':
            self.fgm = FGM(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)
        if self.config.adv_option == 'PGD':
            self.pgd = PGD(self.model, emb_name=self.config.adv_name, epsilon=self.config.adv_epsilon)
        # 混合精度训练
        if self.config.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.config.fp16_opt_level)

        # Train!
        print(">>>>>>>> Running training >>>>>>>>")
        print("  Num examples = %d" %(len(self.train_loader)*self.config.batch_size))
        print("  Num Epochs = %d" %self.config.num_epochs)
        print("  Instantaneous batch size per GPU = %d"%self.config.batch_size)
        print("  GPU ids = %s" %self.config.cuda_visible_devices)
        print("  Total step = %d" %step_total)
        print("  Warm up step = %d" %warmup_steps)
        print("  FP16 Option = %d" %self.config.fp16)
        print(">>>>>>>> Running training >>>>>>>>")

        # step_total = config.num_epochs * len(train_ld)
        step_current = 0
        f1_best = 0
        for epoch in range(self.config.num_epochs):
            progress_bar = ProgressBar(n_total=len(self.train_loader), desc='Training epoch:{0}'.format(epoch))
            for i, batch in enumerate(self.train_loader):
                # 模型推断及计算损失
                self.model.train()
                loss = self.step(batch)
                progress_bar(i, {'loss': loss.item()})
                step_current += 1
                # 模型保存
                if step_current%self.config.step_save==0 and step_current>0:
                    # 模型评估
                    f1_eval = self.eval()
                    # 模型保存
                    f1_best = self.save_checkpoint(step_current, f1_eval, f1_best)
            print('\nEpoch:{0}  Iter:{1}/{2}  loss:{3}\n'.format(epoch, step_current, step_total, loss.item()))


    def step(self, batch):
        """
        每一个batch的训练过程
        """
        # 正常训练
        batch.data = {k:v.to(self.device) for k,v in batch.data.items()}
        outputs = self.model(**batch)
        loss = outputs[0]
        loss = loss.mean()
        # 反向传播
        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # 对抗训练
        if self.config.adv_option == 'FGM':
            self.fgm.attack()
            loss_adv = self.model(**batch)[0]
            if torch.cuda.device_count() > 1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            self.fgm.restore()
        if self.config.adv_option == 'PGD':
            self.pgd.backup_grad()
            K = 3
            for t in range(K):
                self.pgd.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                loss_adv = self.model(**batch)[0]
                loss_adv.backward()                 # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.pgd.restore()                           # 恢复embedding参数
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss


    def save_checkpoint(self, step_current, f1_eval, f1_best):
        """
        模型保存
        """
        if f1_eval != 0:
            path = self.config.path_model_save + 'step_{}/'.format(step_current)
            if not os.path.exists(path):
                os.makedirs(path)
            model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
            model_save.save_pretrained(path)
            print('saving model: {}'.format(path))
            # 保存最优的模型
            if f1_eval > f1_best:
                path = self.config.path_model_save + 'step_best/'
                if not os.path.exists(path):
                    os.makedirs(path)
                model_save = self.model.module if torch.cuda.device_count() > 1 else self.model
                model_save.save_pretrained(path)
                f1_best = f1_eval
                print('saving best model: {}'.format(path))
        return f1_best


    def eval(self):
        """
        评估模型效果
        """
        # 定义metric
        id2label = {i:x for i, x in enumerate(self.tag)}
        label2id = {x:i for i, x in enumerate(self.tag)}
        metric = SpanEntityScore(id2label)
        losses = []
        self.model.eval()
        for i, batch in enumerate(self.valid_loader):
            # 推断
            batch.data = {k:v.to(self.config.device) for k,v in batch.data.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
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
                label = bert_extract_item(start_lab[i], end_lab[i], label2id)
                pred = bert_extract_item(start_pred[i], end_pred[i], label2id)
                metric.update(true_subject=label, pred_subject=pred)
            # label = bert_extract_item(batch.data['label_start'], batch.data['label_end'])
            # pred = bert_extract_item(start_logits, end_logits)
            # metric.update(true_subject=label, pred_subject=pred)
        eval_info, entity_info = metric.result()
        print('\nEval  precision:{0}  recall:{1}  f1:{2}'.format(round(eval_info['acc'],4), round(eval_info['recall'],4), round(eval_info['f1'],4)))
        for item in entity_info.keys():
            print('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}'.format(item, round(entity_info[item]['acc'],4), round(entity_info[item]['recall'],4), round(entity_info[item]['f1'],4)))
        return eval_info['f1']

