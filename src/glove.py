def read_by_tokens(fileobj):

    for line in fileobj:

        for token in line.split():

            yield token


vols = []
vecs = []
vec = []
with open("../data/glove.small.txt") as f:

    tokenized = read_by_tokens(f)
    count = 0
    for token in tokenized:
        if count % 301 == 0:
            vols.append(token)
            if count != 0:
                vecs.append(vec)
                vec = []
        else:
            vec.append(token)
        count += 1

glove_vec = {}
for i in range(0, len(vols) - 1):
    v = vols[i]
    glove_vec[v] = vecs[i]
