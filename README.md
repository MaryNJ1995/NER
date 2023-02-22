
# MarSan at SemEval-2022 Task 11: Multilingual complex named entity recognition using T5 and transformer encoder

The multilingual complex named entity recog-
nition task of SemEval2020 required partici-
pants to detect semantically ambiguous and
complex entities in 11 languages. In order to
participate in this competition, a deep learning
model is being used with the T5 text-to-text lan-
guage model and its multilingual version, MT5,
along with the transformer’s encoder module.
The subtoken check has also been introduced,
resulting in a 4% increase in the model F1-
score in English. We also examined the use of
the BPEmb model for converting input tokens
to representation vectors in this research. A
performance evaluation of the proposed entity
detection model is presented at the end of this
paper. Six different scenarios were defined, and
the proposed model was evaluated in each sce-
nario within the English development set. Our
model is also evaluated in other languages.


![Screenshot from 2023-02-22 13-13-58](https://user-images.githubusercontent.com/86873813/220586905-e83055f8-5c10-4c97-9fa5-e9f7ab59520e.png)

the paper for this implementation can be found [here](https://aclanthology.org/2022.semeval-1.226/)



# Usages
#### Train Model:

> python trainer.py

#### Test Model

> python inferencer.py
