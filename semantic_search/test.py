# %%
import faiss 
import numpy as np

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# %%
data = [ "How do I signup for Autofile in Texas?",      # 0
         "How do I signup for Autofile in Wisconsin?",  # 1
         "How do I signup for Autofile in New Jersey?", # 2
         "Texas Autofile cancellation policy",          # 3
         "Autofile Tax Setup?",                         # 4
         "Texas Toast is great!"                        # 5
         ]                        

sentence_embeddings = model.encode(data)


# %%
dim = 768

index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
articles = np.array(sentence_embeddings)
faiss.normalize_L2(articles)
index.add_with_ids(articles, np.array(range(0, len(articles))))


# Add another article
a = "Testing things"
a = np.array(model.encode([a]))
faiss.normalize_L2(a)

index.add_with_ids(a, np.array([11]))


# %%

n = "Testing things?"
e = np.array(model.encode([n]))
faiss.normalize_L2(e)
index.search(e, 6)


# %%
