import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, n_channel, n_head, n_dim, n_out, activation = "identity"):
        super(MHA, self).__init__()
        activations = {"relu" : nn.ReLU(), "sigmoid" : nn.Sigmoid(), "tanh": nn.Tanh()}
        self.Wq = nn.Parameter(torch.randn(n_head, n_dim, n_channel, dtype = torch.float32))
        self.Wk = nn.Parameter(torch.randn(n_head, n_dim, n_channel, dtype = torch.float32))
        self.Wv = nn.Parameter(torch.randn(n_head, n_dim, n_channel, dtype = torch.float32))
        self.drop = nn.Dropout(p=0.2)
        self.activation = activations[activation] if activation in activations else nn.Identity()
        self.n_channel  = nn.Parameter(torch.tensor(n_channel, dtype = torch.float32), requires_grad = False)
        self.Wo = nn.Parameter(torch.randn(n_head * n_channel, n_out))
        
    def forward(self, X):
        # X is batch x n_seq = 100 x n_dim = 1
        X1 = X.unsqueeze(1)
        q  = torch.matmul(X1, self.Wq) # batch x n_head x n_seq x n_channel
        k  = torch.matmul(X1, self.Wk)
        v  = torch.matmul(X1, self.Wv)
        
        att = F.softmax(torch.matmul(q, torch.transpose(k, 2, 3)) / torch.sqrt(self.n_channel), dim = -1) 
        v = torch.matmul(att, v) # batch x n_head x n_seq x n_channel
        v = self.drop(v)
        vc = torch.concatenate(torch.unbind(v, 1), axis = -1)
        return self.activation(torch.matmul(vc, self.Wo))
        
        
class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.mha1 = MHA(10, 4, 1, 4, activation = "sigmoid")
        self.layer = nn.LayerNorm(4)
        self.mha2 = MHA(5, 3, 4, 1)
    def forward(self, x):
        out = self.mha1(x)
        out = self.layer(out)
        return self.mha2(out)