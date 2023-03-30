import torch
import numpy as np
import json

DUMMY = {'pdb': None, 'seq': '#' * 10,
        'coords': {
            "N": np.zeros((10, 3)) + np.nan,
            "CA":np.zeros((10, 3)) + np.nan,
            "C": np.zeros((10, 3)) + np.nan,
            "O": np.zeros((10, 3)) + np.nan,
        }
}
alphabet = '#ACDEFGHIKLMNPQRSTVWY'  # 0 is padding
HYDROPATHY = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
VOLUME = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
CHARGE = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
POLARITY = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
ACCEPTOR = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
DONOR = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
PMAP = lambda x: [HYDROPATHY[x] / 5, VOLUME[x] / 200, CHARGE[x], POLARITY[x], ACCEPTOR[x], DONOR[x]]


class CDRDataset():

    def __init__(self, jsonl_file, hcdr):
        self.cdrs = []
        self.atgs = []

        with open(jsonl_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                entry = self.get_cdr(lines[i], hcdr)
                if entry is not None and len(entry['seq1']) > 0 and len(entry['seq2']) > 0 and len(entry['seq3']) > 0:
                    self.cdrs.append(entry)
                    self.atgs.append(None)

    def get_cdr(self, s, hcdr):
        entry = json.loads(s)
        seq = entry['seq']
        if seq is None or len(hcdr) == 0:
            return None
        if 'cdr' in entry:
            entry['chain'] = entry['seq']
            entry['seq'] = ''.join(
                [x for x, y in zip(seq, entry['cdr']) if y in hcdr]
            )
        for cdr_type in hcdr:
            if 'cdr' in entry:
                cdr = entry['cdr']
                entry['context' + cdr_type] = ''.join(
                    [(alphabet[0] if y == cdr_type else x) for x, y in zip(seq, cdr)]
                )
                entry['seq' + cdr_type] = ''.join(
                    [x for x, y in zip(seq, cdr) if y == cdr_type]
                )
                cdr_mask = np.array([(y == cdr_type) for y in cdr])
            else:
                cdr_mask = np.array([True] * len(seq))
            entry['coords' + cdr_type] = entry['coords'].copy()
            for key, val in entry['coords'].items():
                val = np.asarray(val)
                val = val[:len(cdr_mask)]
                entry['coords' + cdr_type][key] = val[cdr_mask] if len(cdr_mask) <= len(val) else val

        del entry['coords']
        return entry

    def __len__(self):
        return len(self.cdrs)

    def __getitem__(self, idx):
        return (self.cdrs[idx], self.atgs[idx])

class StructureLoader():

    def __init__(self, dataset, batch_tokens, binder_data=None, interval_sort=0):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        self.binder_data = binder_data

        if interval_sort > 0:
            cdr_type = str(interval_sort)
            self.lengths = [dataset[i]['cdr'].count(cdr_type) for i in range(self.size)]
            self.intervals = [(dataset[i]['cdr'].index(cdr_type), dataset[i]['cdr'].rindex(cdr_type)) for i in range(self.size)]
            sorted_ix = sorted(range(self.size), key=self.intervals.__getitem__)
        else:
            sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_tokens:
                batch.append(ix)
            else:
                clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            if self.binder_data:
                abatch = [self.binder_data[i] for i in b_idx]
                yield (batch, abatch)
            else:
                yield batch

def featurize(batch, context=True):
    loop_length = []
    x_all = []
    s_all = []
    p_all = []
    mask_all = []
    context_all = []

    for i in range(3):
        cdr_length = 0
        for b in batch:
            temp = b['cdr'].count(str(i+1))
            if temp > cdr_length:
                cdr_length = temp
        loop_length.append(cdr_length)

    B = len(batch)
    for j in range(3):
        L_max = max([len(b['seq'+str(j+1)]) for b in batch])
        X = np.zeros([B, L_max, 4, 3])
        S = np.zeros([B, L_max], dtype=np.int32)
        P = np.zeros([B, L_max, 6])
        mask = np.zeros([B, L_max], dtype=np.float32)

        # Build the batch
        for i, b in enumerate(batch):
            x = np.stack([b['coords'+str(j+1)][c] for c in ['N', 'CA', 'C', 'O']], 1)
            if len(x) <= L_max:
                X[i,:len(x),:,:] = x

            l = len(b['seq'+str(j+1)])
            indices = np.asarray([alphabet.index(a) for a in b['seq'+str(j+1)]], dtype=np.int32)
            S[i, :l] = indices
            P[i, :l] = np.array([PMAP(a) for a in b['seq'+str(j+1)]])
            mask[i, :l] = 1.

    # Remove NaN coords
        mask = mask * np.isfinite(np.sum(X,(2,3))).astype(np.float32)
        isnan = np.isnan(X)
        X[isnan] = 0.

        # Conversion
        S = torch.from_numpy(S).long().cuda()
        X = torch.from_numpy(X).float().cuda()
        P = torch.from_numpy(P).float().cuda()
        mask = torch.from_numpy(mask).float().cuda()

        x_all.append(X)
        s_all.append(S)
        p_all.append(P)
        mask_all.append(mask)

        if context:  # extract context
            L_max = max([len(b['context'+str(j+1)]) for b in batch])
            cS = np.zeros([B, L_max], dtype=np.int32)
            cmask = np.zeros([B, L_max], dtype=np.float32)
            crange = [None] * B
            for i, b in enumerate(batch):
                l = len(b['context'+str(j+1)])
                indices = np.asarray([alphabet.index(a) for a in b['context'+str(j+1)]], dtype=np.int32)
                cS[i, :l] = indices
                cmask[i, :l] = 1.
                crange[i] = (b['context'+str(j+1)].index('#'), b['context'+str(j+1)].rindex('#'))

            cmask = torch.from_numpy(cmask).float().cuda()
            cS = torch.from_numpy(cS).long().cuda()
            context = (cS, cmask, crange)
            context_all.append(context)

    X = torch.cat(x_all, dim=1)
    S = torch.cat(s_all, dim=1)
    P = torch.cat(p_all, dim=1)
    mask = torch.cat(mask_all, dim=1)

    return (X, S, P, mask, loop_length), context_all