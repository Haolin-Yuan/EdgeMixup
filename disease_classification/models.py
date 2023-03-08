# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import torch as T
import torch.nn as nn
from torch.nn.functional import one_hot

# Equality of Odds
class AdvNet(nn.Module):
   def __init__(self, num_target_classes, num_protected_classes):
       super(AdvNet, self).__init__()
       self.num_target_classes = num_target_classes
       self.adv = nn.Linear(num_target_classes * 2, num_protected_classes)
   
   def forward(self, y_pred_logits, y_true):
       # concate the true label (one hot) with the logits -> equality of odds instead of demographic parity
       y_true_one_hot = one_hot(y_true, num_classes=self.num_target_classes).float() # B x 4
       adv_feats = T.cat((y_pred_logits, y_true_one_hot), dim=-1) # B x 8
       return self.adv(adv_feats)