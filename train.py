from load import *
import time
import random
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *


def calculate_acc(prob, label):
    # log_prob (N, L), label (N)
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1
    return np.array(acc_train)


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1] - 1
    label = label.view(-1)
    init_label = np.linspace(0, num_label - 1, num_label)
    init_prob = torch.zeros(size=(num_label, num_neg + len(label)), device=prob.device)

    random_ig = random.sample(range(1, l_m + 1), num_neg)
    while len([lab for lab in label if lab in random_ig]) != 0:
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


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):
        return len(self.traj)


class Trainer:
    def __init__(self, model, record):
        self.model = model.to(device)
        self.records = record

        # Ensure loss history keys exist for plotting
        self.records.setdefault('train_loss', [])
        self.records.setdefault('epoch', [])
        self.records.setdefault('acc_valid', [])
        self.records.setdefault('acc_test', [])

        self.start_epoch = record['epoch'][-1] if load and len(record['epoch']) > 0 else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 1
        self.learning_rate = 3e-3
        self.num_epoch = 100
        self.threshold = np.mean(record['acc_valid'][-1]) if load and len(record['acc_valid']) > 0 else 0.0

        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens

        # Cross entropy expects class ids from 0 to C-1
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label - 1, self.len)
        self.data_loader = data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            self.model.train()

            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

            # Track epoch training loss
            epoch_loss_sum = 0.0
            epoch_train_steps = 0

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
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

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)

                    if mask_len <= person_traj_len[0] - 2:
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)

                        epoch_loss_sum += float(loss_train.item())
                        epoch_train_steps += 1

                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:
                        valid_size += person_input.shape[0]
                        acc_valid += calculate_acc(prob, train_label)

                    elif mask_len == person_traj_len[0]:
                        test_size += person_input.shape[0]
                        acc_test += calculate_acc(prob, train_label)

                bar.update(self.batch_size)
            bar.close()

            avg_train_loss = epoch_loss_sum / max(epoch_train_steps, 1)
            acc_valid = np.array(acc_valid) / max(valid_size, 1)
            acc_test = np.array(acc_test) / max(test_size, 1)

            print('epoch:{}, time:{}, train_loss:{:.6f}'.format(
                self.start_epoch + t, time.time() - start, avg_train_loss
            ))
            print('epoch:{}, time:{}, valid_acc:{}'.format(
                self.start_epoch + t, time.time() - start, acc_valid
            ))
            print('epoch:{}, time:{}, test_acc:{}'.format(
                self.start_epoch + t, time.time() - start, acc_test
            ))

            self.records['train_loss'].append(avg_train_loss)
            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + t)

            # Save best checkpoint
            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                torch.save(
                    {
                        'state_dict': self.model.state_dict(),
                        'records': self.records,
                        'time': time.time() - start,
                    },
                    'best_stan_win_1000_' + dname + '.pth'
                )

        # Save final checkpoint with full history
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'records': self.records,
                'time': time.time() - start,
            },
            'final_stan_win_1000_' + dname + '.pth'
        )

    def inference(self):
        user_ids = []
        for t in range(self.num_epoch):
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_valid, cum_test = [0, 0, 0, 0], [0, 0, 0, 0]

            for step, item in enumerate(self.data_loader):
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

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)

                    if mask_len <= person_traj_len[0] - 2:
                        continue
                    elif mask_len == person_traj_len[0] - 1:
                        acc_valid = calculate_acc(prob, train_label)
                        cum_valid += calculate_acc(prob, train_label)
                    elif mask_len == person_traj_len[0]:
                        acc_test = calculate_acc(prob, train_label)
                        cum_test += calculate_acc(prob, train_label)

                print(step, acc_valid, acc_test)

                if acc_valid.sum() == 0 and acc_test.sum() == 0:
                    user_ids.append(step)


if __name__ == '__main__':
    dname = 'NYC'
    file = open('./data/' + dname + '_data.pkl', 'rb')
    file_data = joblib.load(file)

    [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
    mat1 = torch.FloatTensor(mat1)
    mat2s = torch.FloatTensor(mat2s).to(device)
    mat2t = torch.FloatTensor(mat2t)
    lens = torch.LongTensor(lens)

    # Recommended for debugging or partial runs
    part = 100
    trajs, mat1, mat2t, labels, lens = \
        trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]

    ex = (
        mat1[:, :, :, 0].max(),
        mat1[:, :, :, 0].min(),
        mat1[:, :, :, 1].max(),
        mat1[:, :, :, 1].min()
    )

    stan = Model(t_dim=hours + 1, l_dim=l_max + 1, u_dim=u_max + 1, embed_dim=50, ex=ex, dropout=0)

    num_params = 0
    for name in stan.state_dict():
        print(name)

    for param in stan.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    if load:
        checkpoint = torch.load('best_stan_win_1000_' + dname + '.pth', weights_only=False)
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {
            'epoch': [],
            'train_loss': [],
            'acc_valid': [],
            'acc_test': []
        }
        start = time.time()

    trainer = Trainer(stan, records)
    trainer.train()
    # trainer.inference()