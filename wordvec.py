import os
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from tqdm import tqdm
from utils import load_wiener_collections

from gensim.test.utils import get_tmpfile

data_dir = os.path.join('data', 'all_wiener_segmented')

all_docs = load_wiener_collections(data_dir, no_bboxes=True)

# print('training on', num_words, 'number of words')
# model = Doc2Vec(all_docs, vector_size=100, window=2, min_count=1, workers=4, alpha=0.025, min_alpha=0.025, dm=0)
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
# model = Doc2Vec.load(fname)  # you can continue training with the loaded model

model = FastText(all_docs, size=50, window=3, min_count=1, iter=1)
model.save('fasttext.model')
print('hi')