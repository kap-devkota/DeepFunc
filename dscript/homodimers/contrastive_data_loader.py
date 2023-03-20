from typing import Tuple, List
from torch.utils.data import Dataset
from Bio import SeqIO




class TaggedFastaDataset(Dataset):
    def __init__(self, fasta_path: str, tag_path: str):
        self.fastas = []
        self.tags = []
        with open(fasta_path) as f:
            for record in SeqIO.parse(f, "fasta"):
                self.fastas.append(str(record.seq))

        print(self.fastas[0])

        with open(tag_path) as f:
            for tag in f:
                tag = tag.strip()
                if len(tag) > 0:
                    self.tags.append(int(tag))

        assert len(self.fastas) == len(self.tags), "Required the length of tags to be equal to fastas"

    def __len__(self) -> int:
        return len(self.fastas)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.fastas[idx], self.tags[idx]

    def get_proteins(self) -> List[str]:
        return self.fastas



def generate_train_test_split(fasta_path: str, tag_path: str, per_train: float = 0.8) -> Tuple[List[Dataset], Dataset]:
    from torch.utils.data import random_split
    
    fasta_dataset = TaggedFastaDataset(fasta_path, tag_path)

    num_train = int(per_train * len(fasta_dataset))
    num_test = len(fasta_dataset) - num_train
    return random_split(fasta_dataset, [num_train, num_test]), fasta_dataset




