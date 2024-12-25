import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


from pytorch_accelerated.callbacks import TrainerCallback
import torchmetrics

import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import psycopg2
import psycopg2.extras
from settings import DB_RECOMMEND_CONFIG, DB_STATS_CONFIG
from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)

np.random.seed(123)
print(111111111)
register_adapter(np.int64, AsIs)

np.random.seed(123)

'''ratings = pd.read_csv('rating.csv',
                      parse_dates=['timestamp'])

rand_userIds = np.random.choice(ratings['userId'].unique(),
                                size=int(len(ratings['userId'].unique())*0.3),
                                replace=False)

ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

print(ratings.info())'''

#ratings.to_csv("ratings.csv")

ratings = pd.read_csv('rating.csv',
                      parse_dates=['timestamp'])

ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
                                .rank(method='first', ascending=False)

train_ratings = ratings[ratings['rank_latest'] != 1]
#test_ratings = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train_ratings = train_ratings[['userId', 'movieId', 'rating']]
#test_ratings = test_ratings[['userId', 'movieId', 'rating']]

train_ratings.loc[:, 'rating'] = 1

all_movieIds = ratings['movieId'].unique()

class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['userId'], ratings['movieId']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

class DynamicEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices):
        max_index = indices.max().item()
        if max_index >= self.num_embeddings:
            self.expand_embeddings(max_index + 1)
        return self.embeddings(indices)

    def expand_embeddings(self, new_size):
        old_weights = self.embeddings.weight.data
        new_weights = torch.randn(new_size, self.embedding_dim) * 0.01
        new_weights[:self.num_embeddings] = old_weights
        self.num_embeddings = new_size
        self.embeddings = nn.Embedding.from_pretrained(new_weights, freeze=False)


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()
        self.user_embedding = DynamicEmbedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = DynamicEmbedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds

    def forward(self, user_input, item_input):

        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=512, num_workers=2)

    def online_training_step(self, user_input, item_input, labels, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        return loss.item()

"""def online_training(new_data : pd.DataFrame):
  new_user_input = torch.tensor(new_data['userId'].values, dtype=torch.long)
  new_item_input = torch.tensor(new_data['movieId'].values, dtype=torch.long)
  new_labels = torch.tensor(new_data['rating'].values, dtype=torch.float)

  for epoch in range(3):  # Online learning for 3 epochs
    loss = model.online_training_step(new_user_input, new_item_input, new_labels)
    print(f"Online training loss (epoch {epoch + 1}): {loss}")"""

def get_top_k_recommendations_explicit(model, user_ids, all_movie_ids, k=100, start_rank = 0):
    """
    Получить топ-K фильмов для каждого пользователя.

    Args:
        model (NCF): Обученная модель NCF
        user_ids (list): Список ID пользователей
        all_movie_ids (list): Список всех доступных ID фильмов
        k (int): Количество фильмов, которые нужно рекомендовать

    Returns:
        dict: Словарь, где ключи — user_id, а значения — список топ-K ID фильмов
    """
    model.eval()
    recommendations = {}

    # Переводим все фильмы в тензор
    all_movie_tensor = torch.tensor(all_movie_ids, dtype=torch.long)

    for user_id in user_ids:
        # Создаём тензор, где пользователь повторяется для каждого фильма
        user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.long)

        # Предсказания для всех фильмов
        with torch.no_grad():
            scores = model(user_tensor, all_movie_tensor).squeeze()

        # Получаем индексы топ-K фильмов
        top_k_indices = torch.topk(scores, k=k).indices

        # Добавляем в результат соответствующие ID фильмов
        top_k_movie_ids = [(rank + 1 + start_rank, all_movie_ids[i]) for rank, i in enumerate(top_k_indices)]
        recommendations[user_id] = top_k_movie_ids

    return recommendations

num_users = ratings['userId'].max()+1
num_items = ratings['movieId'].max()+1

all_movieIds = ratings['movieId'].unique()

model_ex = NCF(num_users, num_items, train_ratings, all_movieIds)

model_ex.load_state_dict(torch.load("model_weights_dynamic_best.pth", weights_only= True))

#names = pd.read_csv('movie.csv')

ratings_df = ratings

ratings_df["userId"] = ratings_df["userId"].astype(str)

ratings_per_user = ratings_df.groupby('userId').rating.count()
ratings_per_item = ratings_df.groupby('movieId').rating.count()

user_lookup = {v: i+1 for i, v in enumerate(ratings_df['userId'].unique())}
movie_lookup = {v: i+1 for i, v in enumerate(ratings_df['movieId'].unique())}


class UserItemRatingDataset(Dataset):
    def __init__(self, df, movie_lookup, user_lookup):
        self.df = df
        self.movie_lookup = movie_lookup
        self.user_lookup = user_lookup

    def __getitem__(self, index):
        row = self.df.iloc[index]
        user_id = self.user_lookup[row.userId]
        movie_id = self.movie_lookup[row.movieId]

        rating = torch.tensor(row.rating, dtype=torch.float32)

        return (user_id, movie_id), rating

    def __len__(self):
        return len(self.df)

class MfDotBias(nn.Module):

    def __init__(
        self, n_factors, n_users, n_items, ratings_range=None, use_biases=True
    ):
        super().__init__()
        self.bias = use_biases
        self.y_range = ratings_range
        self.user_embedding = nn.Embedding(n_users+1, n_factors, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items+1, n_factors, padding_idx=0)

        if use_biases:
            self.user_bias = nn.Embedding(n_users+1, 1, padding_idx=0)
            self.item_bias = nn.Embedding(n_items+1, 1, padding_idx=0)

    def forward(self, inputs):
        users, items = inputs
        dot = self.user_embedding(users) * self.item_embedding(items)
        result = dot.sum(1)
        if self.bias:
            result = (
                result + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()
            )

        if self.y_range is None:
            return result
        else:
            return (
                torch.sigmoid(result) * (self.y_range[1] - self.y_range[0])
                + self.y_range[0]
            )
        
class RecommenderMetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = torchmetrics.MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "mae": torchmetrics.MeanAbsoluteError(),
            }
        )

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"]
        self.metrics.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()

        mse = metrics["mse"].cpu()
        trainer.run_history.update_metric("mae", metrics["mae"].cpu())
        trainer.run_history.update_metric("mse", mse)
        trainer.run_history.update_metric("rmse",  np.sqrt(mse))

        self.metrics.reset()

def load_mf_model():
    model = MfDotBias(
        120, len(user_lookup), len(movie_lookup), ratings_range=[0.5, 5.5]
    )
    model_path = "best_mf_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Переводим модель в режим оценки
    print(f"Модель загружена из {model_path}")
    return model

all_movies_imp = [movie_lookup[movie_id] for movie_id in ratings_df['movieId'].unique() if movie_id in movie_lookup]

model_imp = load_mf_model()

def recommend_top_k_imp(model, user_ids, all_item_ids, k=30, device='cpu'):
    model.eval()
    model.to(device)
    recommendations = {}

    #max_user_id = model.user_embedding.num_embeddings - 1
    #max_item_id = model.item_embedding.num_embeddings - 1

    #if user_id > max_user_id or user_id < 0:
    #  return []


    # Создаем тензоры для пользователя и всех доступных фильмов
    #user_tensor = torch.tensor([user_id] * len(all_item_ids), dtype=torch.long, device=device)
    item_tensor = torch.tensor(all_item_ids, dtype=torch.long, device=device)

    for user_id in user_ids:
        user_tensor = torch.tensor([user_id] * len(all_item_ids), dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model((user_tensor, item_tensor))

        top_k_indices = torch.topk(scores, k=k).indices

        top_k_item_ids = [(rank + 1, all_item_ids[i]) for rank, i in enumerate(top_k_indices.cpu().numpy())]
        recommendations[user_id] = top_k_item_ids

    return recommendations

"""for user in [0]:
  for movie in rec[user]:
    print(names['title'].loc[names['movieId'] == movie])"""

"""new_data = pd.DataFrame({
    'userId': [4, 5],
    'movieId': [16, 17],
    'rating': [1, 0]
})

online_training(new_data)

rec = get_top_k_recommendations(model, [0, 1, 2], all_movieIds, k = 100)

names = pd.read_csv('movielens-20m-dataset/movie.csv')

for user in [0]:
  for movie in rec[user]:
    print(names['title'].loc[names['movieId'] == movie])"""

def online_training(new_data : pd.DataFrame):
  new_user_input = torch.tensor(new_data['user_id'].values, dtype=torch.long)
  new_item_input = torch.tensor(new_data['movie_id'].values, dtype=torch.long)
  new_labels = torch.tensor(new_data['rating'].values, dtype=torch.float)

  for epoch in range(3):  # Online learning for 3 epochs
    loss = model_ex.online_training_step(new_user_input, new_item_input, new_labels)
    print(f"Online training loss (epoch {epoch + 1}): {loss}")


def get_user_actions():
    query = """SELECT user_id, movie_id FROM user_movie"""
    try:
        with psycopg2.connect(**DB_STATS_CONFIG) as conn:
            df = pd.read_sql(query, con = conn)
    except Exception as e:
        print(f"get_user_actions.Error: {e}")
    
    df.insert(2, 'rating', 1)
    
    return df



#print(get_user_actions())

def recomendations_to_db(rec : dict):
    query = """INSERT INTO user_recommendations(user_id, movie_id)
        VALUES (%(user_id)s, %(movie_id)s);"""
    try:
        with psycopg2.connect(**DB_RECOMMEND_CONFIG) as conn:
            with conn.cursor() as cur:
                for user_id in rec.keys():
                    for movie in rec[user_id]:
                        cur.execute(query, {"user_id": user_id, "movie_id" : movie[1]})
    except Exception as e:
        print(f"recomendations_to_db.Error: {e}")

#recomendations_to_db({1 : [1]})

def get_users() -> list:
    query = """SELECT user_id FROM users"""

    try:
        with psycopg2.connect(**DB_STATS_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query)

                return cur.fetchall()
    except Exception as e:
        print(f"get_users.Error: {e}")


def users_list():
    users_raw = get_users()
    users = []

    for user in users_raw:
        users.append(user[0])
    
    return users
    
#train_part = get_user_actions()

#online_training(train_part)

def update_rec(implicit_k = 3, explicit_k = 3):
    users = users_list()
    #print(users)
    rec1 = recommend_top_k_imp(model_imp, users, all_movies_imp, k = implicit_k)

    rec2 = get_top_k_recommendations_explicit(model_ex, users, all_movieIds, k = explicit_k, start_rank=implicit_k)
    #print(rec)
    recomendations_to_db(rec1)
    recomendations_to_db(rec2)

update_rec()