import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import random
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def k_mers(k, seq):
    if k > len(seq):
        return []
    num = len(seq) - k + 1
    split = []
    for i in range(num):
        split.append(seq[i:i + k])
    return split

def create_tagged_documents(mers, name):
    tagged_docs = [TaggedDocument(mers[i], [str(name[i])]) for i in range(len(name))]
    return tagged_docs

def train_doc2vec_model(mers, name):
    tagged_docs = create_tagged_documents(mers, name)
    model = Doc2Vec(vector_size=100, min_count=1, epochs=100, seed=42)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def get_vector_embeddings(all_mers, all_name, model):
    tagged_docs = create_tagged_documents(all_mers, all_name)
    vectors = {}
    for doc in tagged_docs:
        vectors[doc.tags[0]] = model.infer_vector(doc.words)
    return vectors

set_seed(42)

data = pd.read_excel('target_id_seq.xlsx', header=None, names=['id', 'sequence'])

k = 3
data['k_mers'] = data['sequence'].apply(lambda x: k_mers(k, x))

ids = data['id'].tolist()
k_mers_list = data['k_mers'].tolist()

model = train_doc2vec_model(k_mers_list, ids)

vector_embeddings = get_vector_embeddings(k_mers_list, ids, model)

vector_df = pd.DataFrame.from_dict(vector_embeddings, orient='index')
vector_df.index.name = 'id'
vector_df.reset_index(inplace=True)

vector_df.to_csv('tg_embeddings.csv', index=False)
