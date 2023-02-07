import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch 

class Data(Dataset):
    def __init__(self, matchfile, no_matches, Xa, Xb, nA, nB):
        self.no_matches = no_matches
        self.matchdf = pd.read_csv(matchfile, sep = "\t")
        
        if "scores" in self.matchdf.columns:
            self.matchdf = self.matchdf.sort_values(by = "score", ascending = False).reset_index(drop = True)[: no_matches]
        else:
            self.matchdf = self.matchdf.loc[: no_matches, :]
            # for compatibility
            self.matchdf["scores"] = 1
        self.nA = nA
        self.nB = nB
        self.Xa = Xa
        self.Xb = Xb
        
    def __len__(self):
        return self.no_matches
    
    def __getitem__(self, idx):
        pa, pb, _= self.matchdf.iloc[idx, :].values
        ia, ib = self.nA[pa], self.nB[pb]
        return torch.tensor(self.Xa[ia], dtype = torch.float32).unsqueeze(-1), torch.tensor(self.Xb[ib], dtype = torch.float32).unsqueeze(-1)
        