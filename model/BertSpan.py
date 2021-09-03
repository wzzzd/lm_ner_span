
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

from model.loss.focal_loss import FocalLoss
from model.loss.label_smoothing import LabelSmoothingCrossEntropy




class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, loss_type='ce', soft_label=True):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = soft_label
        self.num_labels = config.num_labels
        self.loss_type = loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()
        

    def forward(self, 
                input_ids, 
                label_start = None,
                label_end = None,
                token_type_ids=None, 
                attention_mask=None,
                max_seq_length=256):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]                                                    # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)                                 # (batch_size, seq_len, hidden_size)
        start_logits = self.start_fc(sequence_output)                                   # (batch_size, seq_len, num_classes)
        if label_start is not None and self.training:
            # 若label_logits长度不等于最大长度，那么填充0
            if input_ids.size()[1] < label_start.size()[1]:
                # size = (label_logits.size()[0], max_seq_length-label_logits.size()[1], label_logits.size()[2])
                # pad_tensor = torch.zeros(size).to(label_start.device)
                # label_logits = torch.cat((label_logits, pad_tensor), 1)
                label_start = label_start[:,:input_ids.size()[1]].contiguous()
                label_end = label_end[:,:input_ids.size()[1]].contiguous()
            
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)  # (batch_size, seq_len, self.num_labels)    # 生成随机矩阵
                label_logits.zero_()                                                    # (batch_size, seq_len, self.num_labels)    # 赋值为0
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, label_start.unsqueeze(2), 1)               # (batch_size, seq_len, self.num_labels)
            else:
                label_logits = label_start.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)                         # (batch_size, seq_len, num_classes)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if label_start is not None and label_end is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)                       # (batch_size * seq_len, num_classes)
            end_logits = end_logits.view(-1, self.num_labels)                           # (batch_size * seq_len, num_classes)
            active_loss = attention_mask.view(-1) == 1                                  # (batch_size * seq_len)                # 只计算非mask的attention值
            active_start_logits = start_logits[active_loss]                             # (batch_size * seq_len, num_classes)
            active_end_logits = end_logits[active_loss]                                 # (batch_size * seq_len, num_classes)

            if input_ids.size()[1] < label_start.size()[1]:
                label_start = label_start[:,:input_ids.size()[1]].contiguous()
                label_end = label_end[:,:input_ids.size()[1]].contiguous()

            active_start_labels = label_start.view(-1)[active_loss]                 # (batch_size * seq_len)
            active_end_labels = label_end.view(-1)[active_loss]                     # (batch_size * seq_len)

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
    
    

class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, label_start=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, label_start], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x
