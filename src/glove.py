import torch

def read_by_tokens(fileobj):
    for line in fileobj:
        for token in line.split():
            yield token

def read_glove(path):
    vols = []
    vecs = []
    vec = []
    with open(path) as f:

        tokenized = read_by_tokens(f)
        count = 0
        for token in tokenized:
            if count % 301 == 0:
                vols.append(token)
                if count != 0:
                    vecs.append(torch.Tensor(vec))
                    vec = []
            else:
                v = torch.tensor(float(token), dtype=torch.float64)
                vec.append(v)
            count += 1
        vecs.append(torch.Tensor(vec))
    f.close()

    glove_vec = {}
    for i in range(0, len(vols)):
        v = vols[i]
        glove_vec[v] = vecs[i]

    return glove_vec
