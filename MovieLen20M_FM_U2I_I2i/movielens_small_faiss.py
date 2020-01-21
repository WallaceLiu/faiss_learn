import faiss
import numpy
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF

# constants
RANDOM_STATE = 0
N_FACTOR = 20
N_RESULT = 10

# load dataset
ratings = numpy.loadtxt(
    'data/ratings.csv',
    delimiter=',',
    skiprows=1,
    usecols=(0, 1, 2),
    dtype=[('userId', 'i8'), ('movieId', 'i8'), ('rating', 'f8')],
)

# rate data
data = ratings['rating']  # list
# user data len(user) is 671
users = sorted(numpy.unique(ratings['userId']))  # list
# movie data len(movies) is 9066
movies = sorted(numpy.unique(ratings['movieId']))  # list

# mapper between id and index
user_id2i = {id: i for i, id in enumerate(users)}
movie_id2i = {id: i for i, id in enumerate(movies)}
movie_i2id = {i: id for i, id in enumerate(movies)}

# matrix row and col
# every value in ratings['userId'] invoke user_id2i.get
row = list(map(user_id2i.get, ratings['userId']))
col = list(map(movie_id2i.get, ratings['movieId']))

# make sparse matrix
rating_matrix = coo_matrix((data, (row, col)))

# non-negative matrix factorization 671*9066 -> 671*20 9066*20
model = NMF(n_components=N_FACTOR, init='random', random_state=RANDOM_STATE)
user_mat = model.fit_transform(rating_matrix)
movie_mat = model.components_.T

# Faiss train
# IndexFlatIP, Exact Search for Inner Product
movie_index = faiss.IndexFlatIP(N_FACTOR)
movie_index.add(movie_mat.astype('float32'))


# search
def users(user_id):
    user_i = user_id2i[user_id]
    user_vec = user_mat[user_i].astype('float32')
    scores, indices = movie_index.search(numpy.array([user_vec]), N_RESULT)
    movie_scores = zip(indices[0], scores[0])
    return [
        {
            "id": int(movie_i2id[i]),
            "score": float(s),
        }
        for i, s in movie_scores
    ]

result = users(1)
print(result)
