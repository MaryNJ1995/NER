from simplerepresentations import RepresentationModel
# from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
# from configuration import BaseConfig
# from flair.data import Sentence
#
# CONFIG_CLASS = BaseConfig()
# CONFIG = CONFIG_CLASS.get_config()
#
#
# def load_data():
#     return ['Hello Transformers!', 'It\'s very simple.']
#
#
# # 4. initialize fine-tuneable transformer embeddings WITH document context
# embeddings = TransformerWordEmbeddings(model=CONFIG.language_model_path,
#                                        layers="-1",
#                                        subtoken_pooling="first",
#                                        fine_tune=True,
#                                        use_context=True,
#                                        )
# flair_forward_embedding = FlairEmbeddings('multi-forward')
# flair_backward_embedding = FlairEmbeddings('multi-backward')
# # text = "سلام احمس من احسانم"
# # sentence = Sentence(text)
# # print(embeddings.embed(sentence))
# # for token in sentence:
# #     print(token)
# #     print(token.embedding.size())

from collections import Counter


def handle_tmp(data):
    while "X" in data:
        data.remove("X")
    while "O" in data:
        data.remove("O")
    c = Counter()
    c.update(data)
    if data:
        return c.most_common(1)[0][0]
    else:
        return "O"


a = ["O", "O", "X", "X", "O", "B-PER", "I-PER"]
b = ["1", "0", "0", "O", "1", "1", "1"]

x = [a[idx] for idx, item in enumerate(b) if item == "1"]

tmp = []
labels = []
for idx, (lbl, sub) in enumerate(zip(a, b)):
    if sub == "1":
        if tmp == []:
            tmp.append(lbl)
        elif tmp != []:
            labels.append(handle_tmp(tmp))
            tmp = []
            tmp.append(lbl)
    else:
        tmp.append(lbl)
if tmp != []:
    labels.append(handle_tmp(tmp))


print(labels)
# print(x)



