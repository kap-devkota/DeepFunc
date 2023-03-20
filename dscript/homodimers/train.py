import sys
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from dscript.models.interaction import ModelInteraction
from dscript.models.embedding import FullyConnectedEmbed 
from dscript.language_model import lm_embed
from dscript.models.contact import ContactCNN
from dscript.utils import load_hdf5_parallel
from dscript.homodimers.contrastive_data_loader import generate_train_test_split
from torch.utils.data import DataLoader


FASTA_PATH = "../../../data/contrastive_data.fasta"
TAG_PATH = "../../../data/contrastive_data.tags"
EMBEDDING_PATH = "../../../data/human_nonRed.h5"


torch.cuda.empty_cache()

def predict_cmap_interaction(model, n0, n1, tensors, use_cuda):
    """
    Predict whether a list of protein pairs will interact, as well as their contact map.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param n0: First protein names
    :type n0: list[str]
    :param n1: Second protein names
    :type n1: list[str]
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool
    """

    b = len(n0)

    p_hat = []
    c_map_mag = []
    y_hat = []
    for i in range(b):
        z_a = lm_embed(n0[i], use_cuda=True)
        z_b = z_a
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        cm, ph, yh = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
        y_hat.append(torch.mean(yh, 1).squeeze(0))
    p_hat = torch.stack(p_hat, 0)
    c_map_mag = torch.stack(c_map_mag, 0)
    y_hat = torch.stack(y_hat, 0)
    return c_map_mag, p_hat, y_hat


def predict_interaction(model, protein: str, tag: int, embeddings):
    c_map_mag, p_hat, y_hat = predict_cmap_interaction(
        model, protein, tag, embeddings, True
    )
    y = tag.cuda()

    p_hat = p_hat.float()
    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())

    loss_func = losses.NTXentLoss()
    return loss_func(y_hat, tag), bce_loss


def train():

    split, dataset = generate_train_test_split(FASTA_PATH, TAG_PATH)
    train_iterator = DataLoader(split[0], shuffle=True, batch_size=10)
    test_iterator = DataLoader(split[1], shuffle=True, batch_size=10)

    proteins = set(dataset.get_proteins())
    embeddings = {} # load_hdf5_parallel(EMBEDDING_PATH, proteins)

    embedding_model = FullyConnectedEmbed(6165, 100, dropout=0.05)
    contact_model = ContactCNN(100, 50, 7)
    linear = torch.nn.Linear(256*256, 20)

    model = ModelInteraction(embedding_model, contact_model, linear,  True)
    params_embedding = [p for p in embedding_model.parameters() if p.requires_grad]
    optim_embedding = torch.optim.Adam(params_embedding, lr=0.001, weight_decay=0)

    params = [p for p in contact_model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=0.001, weight_decay=0)
    model.cuda()

    for epoch in range(10):
        model.train()
        for (p, y) in train_iterator:
            model.zero_grad()
            loss_contrastive, loss_bce = predict_interaction(model, p, y, embeddings)
    
            loss_contrastive.backward(retain_graph=True)
            loss_bce.backward()
            optim.step()
            optim_embedding.step()
    torch.save(model, "homodimer.pt")


if __name__ == "__main__":
    train()
