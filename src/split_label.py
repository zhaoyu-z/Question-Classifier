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
                labels.append(label)
                features.append(feature)
        file.close()
        return features, labels
