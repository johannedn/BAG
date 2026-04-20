import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class SupConLoss(torch.nn.Module):

    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.07, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, x, device, labels=None, dev_score=None, mask=None):
        x = F.normalize(x, p=2, dim=1)
        features = x

        batch_size = features.shape[0]
        ##############
        dev_score = torch.where(dev_score > 5.0, torch.ones_like(dev_score)*5.0, dev_score)
        dev_score = torch.where(dev_score < -5.0, torch.ones_like(dev_score)*(-5.0), dev_score)
        dev_score = dev_score.view(1, -1)
        dev_score_b = (dev_score-dev_score.T).abs()
        mask = torch.where(dev_score_b.abs() <= 1, torch.ones_like(dev_score_b), torch.zeros_like(dev_score_b))
        mask = mask.float().to(device)
        ##############

        # labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float().to(device)

        # print('>>>>>>>>>>>>>>>>>>mask', mask, '\t shape:', mask.shape)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        

        # tile mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        ################
        sim_coeff = 1.0/(1.0+dev_score_b)
        ################

        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = (logits - torch.log(denominator + 1e-6))*sim_coeff
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    


class AnomalyLayer(torch.nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.device = device
        self.vars = nn.ParameterList()
        self.config = [
            ('linear', [dim, dim]),
            ('linear', [1, dim])
        ]
        for i, (name, param) in enumerate(self.config):
            w = nn.Parameter(torch.ones(*param, device=self.device))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(param[0], device=self.device)))

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
        x = F.normalize(x, dim=1)
        idx = 0
        for name, param in self.config:
            w, b = vars[idx], vars[idx + 1]
            x = F.linear(x, w, b)
            idx += 2

        assert idx == len(vars)

        return x

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class GDN(torch.nn.Module):
    def __init__(self,  device):
        super(GDN, self).__init__()
        self.net = AnomalyLayer(172, device)
        self.memory_size = 3000 # 5000 # 4000
        self.sample_size = 1000 # 2000 # 1000
        self.device = device
        self.memory = torch.randn_like(torch.zeros(self.memory_size, dtype=torch.float32, requires_grad=False)).to(device)
        self.time_memory = torch.zeros(self.memory_size, dtype=torch.float32, requires_grad=False).to(device)
        self.idx = 0
        self.fc1 = torch.nn.Linear(1, 1)
        self.suploss = SupConLoss()
  
    def dev_loss(self, y_true, y_prediction, current_time):
        '''
        add time decay
        '''
        index = torch.LongTensor(random.sample(range(self.memory_size), self.sample_size)).to(self.device)
        ref = torch.index_select(self.memory, 0, index)
        ref_time = torch.index_select(self.time_memory, 0, index)

        lambda_t = 1 / 3600
        time_diff = (torch.abs(current_time.reshape([-1, 1]) - ref_time.reshape([1,-1])) )
        w_time = 1 / (torch.log(lambda_t * time_diff +1) +1)
        
        mean_value = torch.sum(torch.mul(w_time, ref), dim=1) / torch.sum(w_time, dim=1)
        std_value =  torch.pow((ref.reshape([1,-1]) - mean_value.reshape([-1, 1])), 2)
        std_value = torch.sum(torch.mul(w_time, std_value), dim=1) / torch.sum(w_time, dim=1)
        std_value = torch.sqrt(std_value)
        dev = (y_prediction - mean_value) / (std_value +1e-6)
        inlier_loss = torch.abs(dev)
        outlier_loss = 5 - torch.abs(dev)
        outlier_loss[outlier_loss < 0.] = 0
        loss = (1 - y_true) * inlier_loss.flatten() + y_true * outlier_loss.flatten()

        return torch.mean(loss)

    def dev_diff(self, y_prediction, current_time, label):
        index = torch.LongTensor(random.sample(range(self.memory_size), self.sample_size)).to(self.device)
        ref = torch.index_select(self.memory, 0, index)
        ref_time = torch.index_select(self.time_memory, 0, index)
        lambda_t = 1 / 3600
        time_diff = (torch.abs(current_time.reshape([-1, 1]) - ref_time.reshape([1, -1])))
        w_time = 1 / (torch.log(lambda_t * time_diff + 1) + 1)

        mean_value = torch.sum(torch.mul(w_time, ref), dim=1) / torch.sum(w_time, dim=1)
        std_value = torch.pow((ref.reshape([1, -1]) - mean_value.reshape([-1, 1])), 2)
        std_value = torch.sum(torch.mul(w_time, std_value), dim=1) / torch.sum(w_time, dim=1)
        std_value = torch.sqrt(std_value)

        dev = (y_prediction - torch.mean(ref)) / torch.std(ref)


        ###multi group
        group = torch.where(dev > -3, torch.ones_like(y_prediction), torch.zeros_like(y_prediction))
        group = torch.where(dev > -2, group+1, group)
        group = torch.where(dev > -1, group + 1, group)
        group = torch.where(dev > 1, group + 1, group)
        group = torch.where(dev > 2, group + 1, group)
        group = torch.where(dev > 3, group + 1, group)

        return dev, group


    def forward(self, x, time, label=None):
        ana_score = self.net(x)
        if self.training:
            record_mem = (ana_score.clone().detach())[label <= 0]
            record_time_mem = (time.clone().detach())[label <= 0]
            self.idx_after = self.idx + record_mem.shape[0]
            if self.idx_after < self.memory_size:
                self.memory[self.idx:self.idx_after] = record_mem.reshape([-1])
                self.time_memory[self.idx:self.idx_after] = record_time_mem.reshape([-1])
            elif self.idx_after >= self.memory_size:
                self.memory[self.idx:self.memory_size] = record_mem.reshape([-1])[0:(self.memory_size - self.idx)]
                self.memory[0:self.idx_after % self.memory_size] = record_mem.reshape([-1])[
                                                                   (self.memory_size - self.idx):]
                self.time_memory[self.idx:self.memory_size] = record_time_mem.reshape([-1])[0:(self.memory_size - self.idx)]
                self.time_memory[0:self.idx_after % self.memory_size] = record_time_mem.reshape([-1])[
                                                                   (self.memory_size - self.idx):]

            self.idx = self.idx_after % self.memory_size

        return ana_score
    
class DA(nn.Module):

    def __init__(self, p=0.5, eps=1e-6):
        super(DA, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1
    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):

        t = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
       
        t = t.repeat(1,x.shape[1])

        return t

    def forward(self, x):
      
        if not self.training:
            return x
   
        mean = x.mean(dim=[-1], keepdim=False)
        std = (x.var(dim=[-1], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)
        beta = self._reparameterize(mean,sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)
        
        # x = (x - mean.reshape(x.shape[0], 1)) / std.reshape(x.shape[0], 1)
        # x_dsu = x * gamma.reshape(x.shape[0], -1) + beta.reshape(x.shape[0], -1)
        x = (x - mean.reshape(x.shape[0],x.shape[1],1)) / std.reshape(x.shape[0],x.shape[1],1)
        x_dsu = x * gamma.reshape(x.shape[0],x.shape[1],1) + beta.reshape(x.shape[0],x.shape[1],1)
      
        return x_dsu
