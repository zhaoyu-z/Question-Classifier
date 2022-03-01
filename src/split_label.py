class SplitLabel:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def generate_sentences(self):
        count = 0
        features = []
        labels = []
        with open(self.corpus_path, 'r', encoding = "ISO-8859-1") as file:
            for line in file:
                sentence = line.replace('\n','')
                [label, feature]= sentence.split(sep=' ', maxsplit=1)
                for sent in sentence:
                    count += 1
                    if (count % 10000==0):
                        print ("read {0} sentences".format (count))
                labels.append(label)
                features.append(feature)
            print ("Done reading file")
            print ()
        file.close()
        return features, labels

# split = SplitLabel("../data/train_5500.label.txt")
# feature, label = split.generate_sentences()
# print(label)
