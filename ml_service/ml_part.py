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
from settings import DB_CONFIG
from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)

np.random.seed(123)

ratings = pd.read_csv('rating.csv',
                      parse_dates=['timestamp'])

rand_userIds = np.random.choice(ratings['userId'].unique(),
                                size=int(len(ratings['userId'].unique())*0.3),
                                replace=False)

ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
                                .rank(method='first', ascending=False)

train_ratings = ratings[ratings['rank_latest'] != 1]
test_ratings = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train_ratings = train_ratings[['userId', 'movieId', 'rating']]
test_ratings = test_ratings[['userId', 'movieId', 'rating']]

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

def get_top_k_recommendations(model, user_ids, all_movie_ids, k=100):
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
        top_k_movie_ids = [(rank + 1, all_movie_ids[i]) for rank, i in enumerate(top_k_indices)]
        recommendations[user_id] = top_k_movie_ids

    return recommendations

num_users = ratings['userId'].max()+1
num_items = ratings['movieId'].max()+1

all_movieIds = ratings['movieId'].unique()

model = NCF(num_users, num_items, train_ratings, all_movieIds)

model.load_state_dict(torch.load("model_weights_dynamic_best.pth"))

#rec = get_top_k_recommendations(model, [0, 46, 1115, 452], all_movieIds, k = 3)

names = pd.read_csv('movie.csv')

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
    loss = model.online_training_step(new_user_input, new_item_input, new_labels)
    print(f"Online training loss (epoch {epoch + 1}): {loss}")

from test import get_user_actions
from test import recomendations_to_db
from test import users_list
#train_part = get_user_actions()

#online_training(train_part)

def update_rec():
    users = users_list()
    rec = get_top_k_recommendations(model, users, all_movieIds, k = 3)
    recomendations_to_db(rec)

update_rec()
#recomendations_to_db(rec)