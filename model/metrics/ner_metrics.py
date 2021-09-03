
import torch
from collections import Counter



class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        # 
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def bert_extract_item(line_start, line_end):
    S = []
    for i, s_l in enumerate(line_start):
        if s_l == 0:
            continue
        for j, e_l in enumerate(line_end[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S



# def bert_extract_item(start_logits, end_logits):
#     S = []
#     # 区分是ground true还是prediction
#     if len(start_logits.size()) == 2:
#         start_pred = start_logits.cpu().numpy()[:,1:-1]
#         end_pred = end_logits.cpu().numpy()[:,1:-1]
#     else:
#         start_pred = torch.argmax(start_logits, -1)[:,1:-1].cpu().numpy()
#         end_pred = torch.argmax(end_logits, -1)[:,1:-1].cpu().numpy()
        
#     for line_start, line_end in zip(start_pred, end_pred):
#         for i, s_l in enumerate(line_start):
#             if s_l == 0:
#                 continue
#             for j, e_l in enumerate(line_end[i:]):
#                 if s_l == e_l:
#                     S.append((s_l, i, i + j))
#                     break
#         # print(1)
        
#     return S



# def bert_extract_item(start_logits, end_logits):
#     S = []
#     # zero = torch.zeros_like(start_logits)
#     # 若start_logits为非全0tensor
#     # if (start_logits!=zero).sum() != 0:
#     if len(start_logits.size()) == 2:
#         start_pred = start_logits.cpu().numpy()[:,1:-1]
#         end_pred = end_logits.cpu().numpy()[:,1:-1]
#         for line_start, line_end in zip(start_pred, end_pred):
#             for i, s_l in enumerate(line_start):
#                 if s_l == 0:
#                     continue
#                 for j, e_l in enumerate(line_end[i:]):
#                     if s_l == e_l:
#                         S.append((s_l, i, i + j))
#                         break
#             # if S != []:
#             #     print(1)
#     else:
#         start_pred = torch.argmax(start_logits, -1)[:,1:-1].cpu().numpy()
#         end_pred = torch.argmax(end_logits, -1)[:,1:-1].cpu().numpy()
#         for line_start, line_end in zip(start_pred, end_pred):
#             for i, s_l in enumerate(line_start):
#                 if s_l == 0:
#                     continue
#                 for j, e_l in enumerate(line_end[i:]):
#                     if s_l == e_l:
#                         S.append((s_l, i, i + j))
#                         break
#         # print(1)
        
#     return S
