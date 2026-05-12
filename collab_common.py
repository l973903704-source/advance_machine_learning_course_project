from load import *
import argparse
import os
import random
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch import nn, optim
from tqdm import tqdm


seed = 0
hours = 24 * 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_seed = 0


def set_seed(seed_value):
    global seed, global_seed
    seed = seed_value
    global_seed = 0
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def to_npy(x):
    return x.detach().cpu().numpy()


class AttnWithCollaborative(nn.Module):
    def __init__(self, emb_loc, loc_max, collab_weight=1.0, c_clip=0.0):
        super().__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max
        self.collab_weight = collab_weight
        self.c_clip = c_clip
        self.track_c_stats = False
        self.last_c_stats = None

    def set_track_c_stats(self, enabled=True):
        self.track_c_stats = enabled

    def pop_c_stats(self):
        stats = self.last_c_stats
        self.last_c_stats = None
        return stats

    def forward(self, self_attn, self_delta, traj_len, collab_bias=None):
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)
        n_batch, n_loc, _ = self_delta.shape
        candidates = torch.arange(1, int(self.loc_max) + 1, device=self_attn.device).long()
        candidates = candidates.unsqueeze(0).expand(n_batch, -1)
        emb_candidates = self.emb_loc(candidates)
        base_score = torch.bmm(emb_candidates, self_attn.transpose(-1, -2))
        attn = base_score * self_delta
        if collab_bias is not None:
            raw_c = collab_bias
            used_c = raw_c
            if self.c_clip and self.c_clip > 0:
                used_c = torch.clamp(raw_c, min=-self.c_clip, max=self.c_clip)
            weighted_c = self.collab_weight * used_c

            if self.track_c_stats:
                with torch.no_grad():
                    base_abs = attn.detach().abs()
                    raw_abs = raw_c.detach().abs()
                    used_abs = used_c.detach().abs()
                    weighted_abs = weighted_c.detach().abs()
                    base_mean = float(base_abs.mean().cpu().item())
                    base_max = float(base_abs.max().cpu().item())
                    weighted_mean = float(weighted_abs.mean().cpu().item())
                    weighted_max = float(weighted_abs.max().cpu().item())
                    eps = 1e-12
                    self.last_c_stats = {
                        'c_raw_mean': float(raw_c.detach().mean().cpu().item()),
                        'c_raw_std': float(raw_c.detach().std(unbiased=False).cpu().item()),
                        'c_raw_abs_mean': float(raw_abs.mean().cpu().item()),
                        'c_raw_abs_max': float(raw_abs.max().cpu().item()),
                        'c_used_abs_mean': float(used_abs.mean().cpu().item()),
                        'c_used_abs_max': float(used_abs.max().cpu().item()),
                        'c_weighted_abs_mean': weighted_mean,
                        'c_weighted_abs_max': weighted_max,
                        'base_abs_mean': base_mean,
                        'base_abs_max': base_max,
                        'c_to_base_mean_ratio': weighted_mean / (base_mean + eps),
                        'c_to_base_max_ratio': weighted_max / (base_max + eps),
                    }
                self.track_c_stats = False

            attn = attn + weighted_c
        attn_out = self.value(attn).view(n_batch, n_loc)
        return attn_out


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size):
        super().__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)
        attn = F.softmax(attn, dim=-1) * mask
        attn_out = torch.bmm(attn, self.value(joint))
        return attn_out


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super().__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, mat2, vec, traj_len):
        delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i]] = 1
            delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i] - 1)[:traj_len[i]])

        esl, esu = self.emb_sl(mask), self.emb_su(mask)
        etl, etu = self.emb_tl(mask), self.emb_tu(mask)
        vsl = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vsu = (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtl = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtu = (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval
        return delta


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super().__init__()
        self.emb_t, self.emb_l, self.emb_u, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        traj = traj.clone()
        traj[:, :, 2] = (traj[:, :, 2] - 1) % hours + 1
        time = self.emb_t(traj[:, :, 2])
        loc = self.emb_l(traj[:, :, 1])
        user = self.emb_u(traj[:, :, 0])
        joint = time + loc + user

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        esl, esu = self.emb_sl(mask), self.emb_su(mask)
        etl, etu = self.emb_tl(mask), self.emb_tu(mask)
        vsl = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vsu = (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtl = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtu = (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval
        return joint, delta


class PrototypeCollaborativeMemory(nn.Module):
    def __init__(self, loc_max, embed_dim, momentum=0.9):
        super().__init__()
        self.loc_max = loc_max
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.register_buffer('prototype_bank', torch.zeros(loc_max, embed_dim))
        self.register_buffer('prototype_count', torch.zeros(loc_max))
        self.eps = 1e-8

    def get_candidate_prototypes(self):
        return self.prototype_bank

    def get_position_bias(self, self_attn, traj_len):
        prototypes = self.get_candidate_prototypes()
        hist = F.normalize(self_attn, p=2, dim=-1)
        proto = F.normalize(prototypes + self.eps, p=2, dim=-1)
        bias = torch.einsum('nmd,ld->nlm', hist, proto)
        mask = torch.zeros_like(bias)
        for i in range(mask.shape[0]):
            mask[i, :, :traj_len[i]] = 1.0
        return bias * mask

    @torch.no_grad()
    def update_memory(self, self_attn, label, traj_len):
        n_batch = self_attn.shape[0]
        for i in range(n_batch):
            target = int(label[i].item())
            valid_len = int(traj_len[i].item())
            if target < 0 or target >= self.loc_max or valid_len <= 0:
                continue
            summary = self_attn[i, :valid_len].mean(dim=0).detach()
            if self.prototype_count[target] == 0:
                self.prototype_bank[target] = summary
            else:
                self.prototype_bank[target] = self.momentum * self.prototype_bank[target] + (1.0 - self.momentum) * summary
            self.prototype_count[target] += 1.0


class NeighborCollaborativeMemory(PrototypeCollaborativeMemory):
    def __init__(self, loc_max, embed_dim, distance_matrix, top_k=16, temperature=1.0, momentum=0.9):
        super().__init__(loc_max, embed_dim, momentum=momentum)
        top_k = min(top_k, loc_max)
        distance_matrix = distance_matrix[:loc_max, :loc_max].detach().float().cpu()
        nearest_distance, nearest_index = torch.topk(distance_matrix, k=top_k, dim=-1, largest=False)
        nearest_weight = F.softmax(-nearest_distance / max(temperature, 1e-6), dim=-1)
        self.register_buffer('neighbor_index', nearest_index.long())
        self.register_buffer('neighbor_weight', nearest_weight.float())

    def get_candidate_prototypes(self):
        neighbor_proto = self.prototype_bank[self.neighbor_index]
        mixed_proto = (self.neighbor_weight.unsqueeze(-1) * neighbor_proto).sum(dim=1)
        return mixed_proto


class ContrastiveCollaborativeMemory(nn.Module):
    def __init__(self, loc_max, embed_dim, temperature=0.1):
        super().__init__()
        self.loc_max = loc_max
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.destination_bank = nn.Parameter(torch.randn(loc_max, embed_dim) * 0.02)

    def project_positions(self, self_attn):
        return F.normalize(self.proj(self_attn), p=2, dim=-1)

    def get_position_bias(self, self_attn, traj_len):
        hist = self.project_positions(self_attn)
        dest = F.normalize(self.destination_bank, p=2, dim=-1)
        bias = torch.einsum('nmd,ld->nlm', hist, dest)
        mask = torch.zeros_like(bias)
        for i in range(mask.shape[0]):
            mask[i, :, :traj_len[i]] = 1.0
        return bias * mask

    def auxiliary_loss(self, self_attn, label, traj_len):
        summary = []
        for i in range(self_attn.shape[0]):
            valid_len = int(traj_len[i].item())
            valid_len = max(valid_len, 1)
            summary.append(self_attn[i, :valid_len].mean(dim=0))
        summary = torch.stack(summary, dim=0)
        query = F.normalize(self.proj(summary), p=2, dim=-1)
        dest = F.normalize(self.destination_bank, p=2, dim=-1)
        logits = torch.matmul(query, dest.transpose(0, 1)) / self.temperature
        return F.cross_entropy(logits, label.view(-1))

    def update_memory(self, self_attn, label, traj_len):
        return None


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        self.traj = traj
        self.mat1 = m1
        self.vec = v
        self.label = label
        self.length = length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):
        return len(self.traj)


class BaseCollaborativeSTAN(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, collaborative_module, collab_weight=1.0, c_clip=0.0):
        super().__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim - 1, embed_layers)
        self.Collab = collaborative_module
        self.Attn = AttnWithCollaborative(emb_l, l_dim - 1, collab_weight=collab_weight, c_clip=c_clip)

    def encode_history(self, traj, mat1, traj_len):
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)
        self_attn = self.SelfAttn(joint, delta, traj_len)
        return self_attn

    def forward(self, traj, mat1, mat2, vec, traj_len, return_hidden=False):
        self_attn = self.encode_history(traj, mat1, traj_len)
        self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)
        collab_bias = self.Collab.get_position_bias(self_attn, traj_len)
        output = self.Attn(self_attn, self_delta, traj_len, collab_bias=collab_bias)
        if return_hidden:
            return output, self_attn
        return output

    @torch.no_grad()
    def update_collaborative_memory(self, self_attn, label, traj_len):
        self.Collab.update_memory(self_attn.detach(), label.detach(), traj_len.detach())

    def get_auxiliary_loss(self, self_attn, label, traj_len):
        if hasattr(self.Collab, 'auxiliary_loss'):
            return self.Collab.auxiliary_loss(self_attn, label, traj_len)
        return None

    def enable_collab_stats(self, enabled=True):
        self.Attn.set_track_c_stats(enabled)

    def pop_collab_stats(self):
        return self.Attn.pop_c_stats()


class PrototypeSTAN(BaseCollaborativeSTAN):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, collab_weight=1.0, momentum=0.9, c_clip=0.0):
        collab = PrototypeCollaborativeMemory(loc_max=l_dim - 1, embed_dim=embed_dim, momentum=momentum)
        super().__init__(t_dim, l_dim, u_dim, embed_dim, ex, collab, collab_weight=collab_weight, c_clip=c_clip)


class NeighborSTAN(BaseCollaborativeSTAN):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, distance_matrix, collab_weight=1.0, top_k=16, temperature=1.0, momentum=0.9, c_clip=0.0):
        collab = NeighborCollaborativeMemory(
            loc_max=l_dim - 1,
            embed_dim=embed_dim,
            distance_matrix=distance_matrix,
            top_k=top_k,
            temperature=temperature,
            momentum=momentum,
        )
        super().__init__(t_dim, l_dim, u_dim, embed_dim, ex, collab, collab_weight=collab_weight, c_clip=c_clip)


class ContrastiveSTAN(BaseCollaborativeSTAN):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, collab_weight=1.0, temperature=0.1, c_clip=0.0):
        collab = ContrastiveCollaborativeMemory(loc_max=l_dim - 1, embed_dim=embed_dim, temperature=temperature)
        super().__init__(t_dim, l_dim, u_dim, embed_dim, ex, collab, collab_weight=collab_weight, c_clip=c_clip)


def calculate_acc(prob, label):
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        pred_np = to_npy(topk_predict_batch)
        label_np = to_npy(label)
        for j, topk_predict in enumerate(pred_np):
            if label_np[j] in topk_predict:
                acc_train[i] += 1
    return np.array(acc_train)


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1] - 1
    label = label.view(-1)
    init_label = np.linspace(0, num_label - 1, num_label)
    init_prob = torch.zeros(size=(num_label, num_neg + len(label)), device=prob.device)

    random_ig = random.sample(range(1, l_m + 1), num_neg)
    while len([lab for lab in label if int(lab.item()) in random_ig]) != 0:
        random_ig = random.sample(range(1, l_m + 1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i - len(label)]]

    return init_prob.float(), torch.LongTensor(init_label).to(prob.device)


class CollaborativeTrainer:
    def __init__(self, model, record, trajs, mat1, mat2s, mat2t, labels, lens, dname, part, load_flag=False,
                 num_neg=10, batch_size=1, learning_rate=3e-3, num_epoch=100, aux_weight=0.1,
                 weight_decay=0.0, lr_step=1000, lr_gamma=1.0, best_metric='r5',
                 early_stop_patience=0, log_c_every=0, warn_c_abs=2.0, warn_c_ratio=2.0):
        self.model = model.to(device)
        self.records = record
        self.records.setdefault('train_loss', [])
        self.records.setdefault('train_ce_loss', [])
        self.records.setdefault('train_aux_loss', [])
        self.records.setdefault('c_stats', [])
        self.start_epoch = record['epoch'][-1] if load_flag and len(record['epoch']) > 0 else 1
        self.num_neg = num_neg
        self.batch_size = batch_size
        if self.batch_size != 1:
            raise ValueError('The current STAN trajectory loop supports only batch_size=1.')
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.aux_weight = aux_weight
        self.weight_decay = weight_decay
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.best_metric = best_metric
        self.early_stop_patience = early_stop_patience
        self.log_c_every = log_c_every
        self.warn_c_abs = warn_c_abs
        self.warn_c_ratio = warn_c_ratio
        self.global_forward_step = 0
        self.threshold = self._metric_score(record['acc_valid'][-1]) if load_flag and len(record['acc_valid']) > 0 else -np.inf
        self.traj = trajs
        self.mat1 = mat1
        self.mat2s = mat2s
        self.mat2t = mat2t
        self.label = labels
        self.len = lens
        self.dname = dname
        self.part = part
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label - 1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def _metric_score(self, acc):
        metric_index = {'r1': 0, 'r5': 1, 'r10': 2, 'r20': 3}
        if self.best_metric == 'mean':
            return float(np.mean(acc))
        return float(acc[metric_index[self.best_metric]])

    def _prepare_c_stats(self):
        should_log = self.log_c_every > 0 and self.global_forward_step % self.log_c_every == 0
        if should_log:
            self.model.enable_collab_stats(True)
        return should_log

    def _finish_c_stats(self, should_log, split, epoch, mask_len):
        if should_log:
            stats = self.model.pop_collab_stats()
            if stats is not None:
                stats = dict(stats)
                stats.update({
                    'epoch': int(self.start_epoch + epoch),
                    'forward_step': int(self.global_forward_step),
                    'split': split,
                    'mask_len': int(mask_len),
                })
                self.records['c_stats'].append(stats)
                print(
                    '[C-STATS] epoch:{epoch}, fstep:{forward_step}, split:{split}, mask_len:{mask_len}, '
                    'c_abs_mean:{c_raw_abs_mean:.6f}, c_abs_max:{c_raw_abs_max:.6f}, '
                    'weighted_c_abs_mean:{c_weighted_abs_mean:.6f}, base_abs_mean:{base_abs_mean:.6f}, '
                    'weighted/base_mean:{c_to_base_mean_ratio:.6f}'.format(**stats)
                )
                if stats['c_raw_abs_max'] > self.warn_c_abs or stats['c_to_base_mean_ratio'] > self.warn_c_ratio:
                    print(
                        '[C-WARN] C may dominate the STAN score: '
                        'c_abs_max={:.6f}, weighted/base_mean={:.6f}'.format(
                            stats['c_raw_abs_max'], stats['c_to_base_mean_ratio']
                        )
                    )
        self.global_forward_step += 1

    def train(self, ckpt_prefix='best_collab_stan'):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gamma)

        start_time = time.time()
        no_improve_epochs = 0
        for epoch in range(self.num_epoch):
            self.model.train()
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

            epoch_loss_sum = 0.0
            epoch_ce_sum = 0.0
            epoch_aux_sum = 0.0
            epoch_train_steps = 0

            bar = tqdm(total=self.part)
            for _, item in enumerate(self.data_loader):
                person_input, person_m1, person_m2t, person_label, person_traj_len = item
                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long, device=device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32, device=device)

                for mask_len in range(1, person_traj_len[0] + 1):
                    input_mask[:, :mask_len] = 1
                    m1_mask[:, :mask_len, :mask_len] = 1

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long, device=device) + mask_len

                    if mask_len <= person_traj_len[0] - 2:
                        log_c = self._prepare_c_stats()
                        prob, self_attn = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len, return_hidden=True)
                        self._finish_c_stats(log_c, 'train', epoch, mask_len)
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)

                        ce_loss = F.cross_entropy(prob_sample, label_sample)
                        aux_loss = self.model.get_auxiliary_loss(self_attn, train_label, train_len)

                        if aux_loss is not None:
                            loss = ce_loss + self.aux_weight * aux_loss
                            epoch_aux_sum += float(aux_loss.item())
                        else:
                            loss = ce_loss

                        epoch_loss_sum += float(loss.item())
                        epoch_ce_sum += float(ce_loss.item())
                        epoch_train_steps += 1

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        self.model.update_collaborative_memory(self_attn, train_label, train_len)
                    else:
                        with torch.no_grad():
                            split = 'valid' if mask_len == person_traj_len[0] - 1 else 'test'
                            log_c = self._prepare_c_stats()
                            prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)
                            self._finish_c_stats(log_c, split, epoch, mask_len)
                        if mask_len == person_traj_len[0] - 1:
                            valid_size += person_input.shape[0]
                            acc_valid += calculate_acc(prob, train_label)
                        elif mask_len == person_traj_len[0]:
                            test_size += person_input.shape[0]
                            acc_test += calculate_acc(prob, train_label)

                bar.update(self.batch_size)
            bar.close()

            acc_valid = np.array(acc_valid) / max(valid_size, 1)
            acc_test = np.array(acc_test) / max(test_size, 1)

            avg_train_loss = epoch_loss_sum / max(epoch_train_steps, 1)
            avg_train_ce_loss = epoch_ce_sum / max(epoch_train_steps, 1)
            avg_train_aux_loss = epoch_aux_sum / max(epoch_train_steps, 1)

            print(f'epoch:{self.start_epoch + epoch}, time:{time.time() - start_time}, train_loss:{avg_train_loss:.6f}, train_ce_loss:{avg_train_ce_loss:.6f}, train_aux_loss:{avg_train_aux_loss:.6f}')
            print(f'epoch:{self.start_epoch + epoch}, time:{time.time() - start_time}, valid_acc:{acc_valid}')
            print(f'epoch:{self.start_epoch + epoch}, time:{time.time() - start_time}, test_acc:{acc_test}')

            self.records['train_loss'].append(avg_train_loss)
            self.records['train_ce_loss'].append(avg_train_ce_loss)
            self.records['train_aux_loss'].append(avg_train_aux_loss)
            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + epoch)

            valid_score = self._metric_score(acc_valid)
            if self.threshold < valid_score:
                self.threshold = valid_score
                no_improve_epochs = 0
                save_path = f'{ckpt_prefix}_{self.dname}.pth'
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'records': self.records,
                    'time': time.time() - start_time,
                }, save_path)
                print(f'best_{self.best_metric}:{valid_score:.6f}, saved:{save_path}')
            else:
                no_improve_epochs += 1
                if self.early_stop_patience > 0 and no_improve_epochs >= self.early_stop_patience:
                    print(f'early stop: no valid {self.best_metric} improvement for {no_improve_epochs} epochs')
                    break


def load_dataset(dname, part):
    with open(f'./data/{dname}_data.pkl', 'rb') as file:
        file_data = joblib.load(file)
    trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max = file_data
    mat1 = torch.FloatTensor(mat1)
    mat2s = torch.FloatTensor(mat2s).to(device)
    mat2t = torch.FloatTensor(mat2t)
    lens = torch.LongTensor(lens)

    if part is None or part <= 0 or part > len(lens):
        part = len(lens)

    trajs = trajs[:part]
    mat1 = mat1[:part]
    mat2t = mat2t[:part]
    labels = labels[:part]
    lens = lens[:part]

    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
    return trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max, ex


def build_arg_parser(default_variant):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', type=str, default='NYC')
    parser.add_argument('--part', type=int, default=0, help='Number of users to run. Use 0 for all users.')
    parser.add_argument('--embed_dim', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--num_neg', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--collab_weight', type=float, default=1.0)
    parser.add_argument('--c_clip', type=float, default=0.0, help='Clamp C to [-c_clip, c_clip]. Use 0 to disable.')
    parser.add_argument('--aux_weight', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_step', type=int, default=1000)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--best_metric', type=str, default='r5', choices=['r1', 'r5', 'r10', 'r20', 'mean'])
    parser.add_argument('--early_stop_patience', type=int, default=0, help='0 disables early stopping.')
    parser.add_argument('--log_c_every', type=int, default=1000, help='Log C statistics every N forward calls. 0 disables.')
    parser.add_argument('--warn_c_abs', type=float, default=2.0)
    parser.add_argument('--warn_c_ratio', type=float, default=2.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument('--neighbor_temperature', type=float, default=1.0)
    parser.add_argument('--contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--variant', type=str, default=default_variant)
    parser.add_argument('--load_ckpt', action='store_true')
    parser.add_argument('--save_prefix', type=str, default=f'best_{default_variant}_stan')
    return parser
