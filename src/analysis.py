pred = open("fa.pred.conll")
dev = open("../data/SemEval2022-Task11_Train-Dev/FA-Farsi/fa_dev.conll")

sentences, labels, tokens, tags = [], [], [], []

for line in dev:
    if not line.startswith("# id"):
        if line == "\n":
            sentences.append(tokens)
            labels.append(tags)
            tokens, tags = [], []
        else:
            line = line.strip().split()
            tokens.append(line[0].strip())
            tags.append(line[3].strip())

labels_pred, tags_pred = [], []
for line in pred:
    if not line.startswith("# id"):
        if line == "\n":
            labels_pred.append(tags_pred)
            tags_pred = []
        else:
            tags_pred.append(line.strip())


assert len(labels) == len(labels_pred)
assert len(tags) == len(tags_pred)

outputs = open("analysis.txt", "w")
# print(sentences)
# print(labels)
print(labels_pred)


for token, label, pred_tags in zip(sentences, labels, labels_pred):
    for t, l, p in zip(token, label, pred_tags):
        outputs.write(str(t) + " " + str(l) + " " + str(p))
        outputs.write("\n")

    outputs.write("\n")