import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint as odeint
from time import time
from utils import *
from modules import MLP


class EncoderFunc(nn.Module):
    def __init__(self, num_atoms, v_dim, e_dim):
        super().__init__()

        self.num_atoms = num_atoms
        self.mlp_v = MLP(n_in=e_dim, n_hid=32, n_out=v_dim) #nn.Sequential(nn.Linear(e_dim, 16),
                     #              nn.Tanh(),
                     #              nn.Linear(16, v_dim))
        self.mlp_e = MLP(n_in=v_dim*2, n_hid=32, n_out=e_dim)#nn.Sequential(nn.Linear(v_dim*2, 16),
                     #              nn.Tanh(),
                     #              nn.Linear(16, e_dim))
        off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def edge2node(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(self.rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    """
    y: (v0, e0)
    v0: [num_sims, num_atoms, v_dim]
    e0: [num_sims, num_atoms, num_atoms-1, e_dim]
    returns (with odeint): 
     (dv, de) where
     dv: [num_sims, num_timesteps, num_atoms, v_dim]
     de: [num_sims, num_timesteps, num_atoms, num_atoms-1, e_dim]
    """
    def forward(self, t, y):
        v0, e0 = y
        e0 = e0.view(e0.shape[0], e0.shape[1]*e0.shape[2], e0.shape[3])  # collapse connectivity
        dv = self.mlp_v(self.edge2node(e0))
        de = self.mlp_e(self.node2edge(v0))
        de = de.view(de.shape[0], self.num_atoms, self.num_atoms-1, de.shape[2])
        return dv, de


class DecoderFunc(nn.Module):
    def __init__(self, num_atoms, v_dim, e_dim, num_edge_types):
        super().__init__()

        self.num_atoms = num_atoms
        self.num_edge_types = num_edge_types
        self.mlp_v = MLP(n_in=e_dim, n_hid=64, n_out=v_dim)
        self.mlp_e = nn.ModuleList(
            [MLP(n_in=v_dim*2, n_hid=64, n_out=e_dim) for _ in range(num_edge_types)])
        off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def edge2node(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(self.rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    """
    y: (v0, e0, edge_type)
    v0: [num_sims, num_atoms, v_dim]
    e0: [num_sims, num_atoms, num_atoms-1, e_dim]
    returns (with odeint): 
     (dv, de) where
     dv: [num_sims, num_timesteps, num_atoms, v_dim]
     de: [num_sims, num_timesteps, num_atoms, num_atoms-1, e_dim]
    """
    def forward(self, t, y):
        v0, e0, edge_type = y
        e0 = e0.view(e0.shape[0], e0.shape[1]*e0.shape[2], e0.shape[3])  # collapse connectivity
        dv = self.mlp_v(self.edge2node(e0))
        de = torch.zeros([e0.shape[0], e0.shape[1], e0.shape[2], self.num_edge_types], requires_grad=True).cuda()
        for i in range(self.num_edge_types):
            et = edge_type[:, :, i].unsqueeze(-1)
            de[:, :, :, i] = self.mlp_e[i](self.node2edge(v0)) * et
        de = de.sum(-1).view(de.shape[0], self.num_atoms, self.num_atoms-1, de.shape[2])
        return dv, de, edge_type*0

def main():
    SUFFIX = "_springs5"
    batch_size = 16
    num_atoms = 5
    enc_v_dim = 49 * 4
    enc_e_dim = 2
    dec_v_dim = 4
    dec_e_dim = 4
    num_edge_types = 2
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
        batch_size, SUFFIX)

    encoder = EncoderFunc(num_atoms=num_atoms, v_dim=enc_v_dim, e_dim=enc_e_dim).cuda()
    decoder = DecoderFunc(num_atoms=num_atoms, v_dim=dec_v_dim, e_dim=dec_e_dim, num_edge_types=num_edge_types).cuda()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-1)

    for epoch in range(100):
        t = time()
        train_loss = []
        train_acc = []
        train_mse = []
        for batch_idx, (data, relations) in enumerate(train_loader):
            rel0 = torch.zeros(relations.shape[0], num_atoms, num_atoms - 1, enc_e_dim).cuda()
            edata = data.view(data.shape[0], data.shape[1], -1).cuda()
            relations = relations.view(relations.shape[0], num_atoms, num_atoms-1).cuda()

            # run encoder
            _, e = odeint(encoder, (edata, rel0), torch.Tensor([0, 0.1]).cuda(), atol=1e-7, rtol=1e-4, method='rk4')

            # sample graph
            logits = e[1].view(e[1].shape[0], e[1].shape[1]*e[1].shape[2], e[1].shape[3])
            edges = gumbel_softmax(logits, tau=0.5, hard=False).cuda()
            prob = my_softmax(logits, -1)

            v0 = data[:, :, 0, :].cuda()
            e0 = torch.zeros(relations.shape[0], num_atoms, num_atoms - 1, dec_e_dim).cuda()
            v, _, _ = odeint(decoder, (v0, e0, edges), torch.linspace(0, 0.5, 49).cuda(), atol=1e-7, rtol=1e-4, method='rk4')
            preds = v[1:, :, :, :].permute(1, 2, 0, 3)

            target = data[:, :, 1:, :].cuda()
            loss_nll = nll_gaussian(preds, target, 1e-5)

            loss = loss_nll
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            train_acc.append(edge_accuracy(logits, relations))
            train_mse.append(F.mse_loss(preds, target).item())
        print("Epoch: {:04d}".format(epoch),
              "Train loss: {:.5f}".format(np.mean(train_loss)),
              "Train acc: {:.5f}".format(np.mean(train_acc)),
              'Train mse: {:.5f}'.format(np.mean(train_mse)),
              'Time: {:.2f}s'.format(time() - t))


if __name__ == '__main__':
    main()