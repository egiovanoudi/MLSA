import torch
import random
import argparse
import os

from structgen import *
from tqdm import tqdm


def build_model(args):
    return RevisionDecoder(args).cuda()

def evaluate(model, loader, args):
    model.eval()
    rmsd = []
    val_rmsd1 = []
    val_rmsd2 = []
    val_rmsd3 = []

    with torch.no_grad():
        for hbatch, abatch in tqdm(loader):
            for b in hbatch:
                b_list = []
                b_list.append(b)
                (hX, hS, hL, hmask, task), context = featurize(b_list)
                L = hmask.sum().long().item()
                if L > 0:
                    out = model.log_prob(hS, hmask, task, context=context)
                    X_pred = out.X_cdr
                    x_tasks = torch.split(X_pred, task, dim=1)
                    hx_tasks = torch.split(hX, task, dim=1)
                    hmask_tasks = torch.split(hmask, task, dim=1)

                    for j in range(3):
                        rmsd.append(compute_rmsd(x_tasks[j][:, :, 1, :], hx_tasks[j][:, :, 1, :], hmask_tasks[j]))  # alpha carbon

                    val_rmsd1.append(rmsd[0].item())
                    val_rmsd2.append(rmsd[1].item())
                    val_rmsd3.append(rmsd[2].item())
                    rmsd.clear()

    return sum(val_rmsd1) / len(val_rmsd1), sum(val_rmsd2) / len(val_rmsd2), sum(val_rmsd3) / len(val_rmsd3)


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='dataset/train.jsonl')
parser.add_argument('--val_path', default='dataset/val.jsonl')
parser.add_argument('--test_path', default='dataset/test.jsonl')

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=100)
parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=21)
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--print_iter', type=int, default=50)

args = parser.parse_args()
args.context = True
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

loaders = []
for path in [args.train_path, args.val_path, args.test_path]:
    data = CDRDataset(path, '123')
    loader = StructureLoader(data.cdrs, batch_tokens=args.batch_tokens, binder_data=data.atgs)
    loaders.append(loader)

loader_train, loader_val, loader_test = loaders
model = build_model(args)
optimizer = torch.optim.Adam(model.parameters())
print('Training:{}, Validation:{}, Test:{}'.format(
    len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
)

for e in range(args.epochs):
    model.train()
    meter = 0

    for i, (hbatch,abatch) in enumerate(tqdm(loader_train)):
        optimizer.zero_grad()
        hchain, context = featurize(hbatch)
        if hchain[3].sum().item() == 0:
            continue
        loss = model(*hchain, context=context)
        loss.backward()
        optimizer.step()
        meter += loss.item()
        if (i + 1) % args.print_iter == 0:
            meter /= args.print_iter
            print(f'[{i + 1}] Train Loss = {meter:.4f}')
            meter = 0

    val_rmsd1, val_rmsd2, val_rmsd3 = evaluate(model, loader_val, args)
    print(f'Epoch {e}, Val RMSD1 = {val_rmsd1:.4f}, Val RMSD2 = {val_rmsd2:.4f}, Val RMSD3 = {val_rmsd3:.4f}')

test_rmsd1, test_rmsd2, test_rmsd3 = evaluate(model, loader_test, args)
print(f'Test RMSD1 = {test_rmsd1:.4f}, Test RMSD2 = {test_rmsd2:.4f}, Test RMSD3 = {test_rmsd3:.4f}')