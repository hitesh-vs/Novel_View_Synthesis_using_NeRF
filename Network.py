import torch
import numpy
import torch.nn as nn


class NeRFNetwork(nn.Module):

    def __init__(self,pos_dim=10,dir_dim=4,hidden_layer_dim=256): # L = 10 for position, L = 4 for direction (Factor by which embedding inc dim)
        super(NeRFNetwork, self).__init__()

        # The x6 here is cos each input has 3 dim (xyz), +3 cos we are adding the og input (skip connection)
        self.block1 = nn.Sequential(nn.Linear(pos_dim * 6 + 3, hidden_layer_dim), nn.ReLU(), 
                                    nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(),
                                    nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(),
                                    nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(), )
        
         # density estimation, same reason for the dimensions as above
        self.block2 = nn.Sequential(nn.Linear(pos_dim * 6 + hidden_layer_dim + 3, hidden_layer_dim), nn.ReLU(),
                                    nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(),
                                    nn.Linear(hidden_layer_dim, hidden_layer_dim), nn.ReLU(),
                                    nn.Linear(hidden_layer_dim, hidden_layer_dim + 1), ) # +1 at the end to collect density output
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(dir_dim * 6 + hidden_layer_dim + 3, hidden_layer_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_layer_dim // 2, 3), nn.Sigmoid(), )

        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.pos_dim) # emb_x: [batch_size, pos_dim * 6]
        emb_d = self.positional_encoding(d, self.dir_dim) # emb_d: [batch_size, dir_dim * 6]
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma
    

