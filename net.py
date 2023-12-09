import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dim
        # self.thickness_sup = params.thickness_sup
        # self.N_layers = params.N_layers
        # self.M_materials = params.M_materials
        # self.n_database = params.n_database.view(1, 1, params.M_materials, -1).cuda() # 1 x 1 x number of mat x number of freq

        # self.FC = nn.Sequential(
        #     nn.Linear(self.noise_dim, self.N_layers * (self.M_materials + 1)),
        #     nn.BatchNorm1d(self.N_layers * (self.M_materials + 1))
        # )


    # def forward(self, noise, alpha):
    #     net = self.FC(noise)
    #     net = net.view(-1, self.N_layers, self.M_materials + 1)
        
    #     thicknesses = torch.sigmoid(net[:, :, 0]) * self.thickness_sup
    #     X = net[:, :, 1:]
        
    #     P = F.softmax(X * alpha, dim = 2).unsqueeze(-1) # batch size x number of layer x number of mat x 1
    #     refractive_indices = torch.sum(P * self.n_database, dim=2) # batch size x number of layer x number of freq
        
    #     return (thicknesses, refractive_indices, P.squeeze())

        

class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, dim=16):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(dim, dim*2, bias=False),
                nn.BatchNorm1d(dim*2),
                nn.LeakyReLU(0.2),
                nn.Linear(dim*2, dim, bias=False),
                nn.BatchNorm1d(dim))

    def forward(self, x):
        return F.leaky_relu(self.block(x) + x, 0.2)

'''
class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, dim=64):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2))

    def forward(self, x):
        return x + self.block(x)
'''

class ResGenerator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dim
        self.res_layers = params.res_layers
        self.res_dim = params.res_dim
        self.nums = params.nums
        self.size = self.nums.shape[0]
        
        self.initBLOCK = nn.Sequential(
            nn.Linear(self.noise_dim, self.res_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2)
        )

        self.endBLOCK = nn.Sequential(
            nn.Linear(self.res_dim, self.size * 3, bias=False),
            nn.BatchNorm1d(self.size * 3),
        )

        self.ResBLOCK = nn.ModuleList()
        for i in range(params.res_layers):
            self.ResBLOCK.append(ResBlock(self.res_dim))

        self.FC_bin = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
        )

    def forward(self, noise, alpha):
        net = self.initBLOCK(noise)
        for i in range(self.res_layers):
            self.ResBLOCK[i](net)
        net = self.endBLOCK(net)

        net = net.view(-1, 3, self.size)
        
        X = net[:, 1:, :]
        P = F.softmax(X * alpha, dim=2)
        # num_layers = torch.sum(P * nums, dim=2)
        num_layers = self.nums[P.argmax(dim=2)]
        Bin = torch.sigmoid(self.FC_bin(torch.sum(net[:, 0, :], dim=1).unsqueeze(-1)))
        
        # P = F.softmax(X * alpha, dim = 1).unsqueeze(-1) # batch size x number of layer x number of mat x 1
        # P = (X * alpha).unsqueeze(-1) # batch size x number of layer x number of mat x 1
        # refractive_indices = torch.sum(P * self.n_database, dim=1) # batch size x number of layer x number of freq
        # print(X[:,0].size())
        
        # return (thicknesses, refractive_indices, P.squeeze())
        return (num_layers[:,0], num_layers[:,1], P, Bin)



