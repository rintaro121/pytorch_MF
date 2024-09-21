import os

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn import model_selection

dataset_dir = "./datasets/ml-1m/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MovilensDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = df.user_id.values
        self.items = df.movie_id.values
        self.ratings = df.rating.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        movie_id = self.items[idx]
        ratings = self.ratings[idx]

        return {
            "users": torch.tensor(user_id, dtype=torch.long),
            "movies": torch.tensor(movie_id, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, num_factors)
        self.item_emb = nn.Embedding(num_items, num_factors)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        user_feats = self.user_emb(user_id)
        item_feats = self.item_emb(item_id)

        user_bias = self.user_bias(user_id)
        item_bias = self.item_bias(item_id)

        outputs = (
            (user_feats * item_feats).sum(1) + torch.squeeze(user_bias) + torch.squeeze(item_bias)
        )

        return outputs


if __name__ == "__main__":

    df_ratings = pd.read_csv(
        os.path.join(dataset_dir, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    df_movies = pd.read_csv(
        os.path.join(dataset_dir, "movies.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    le_user = preprocessing.LabelEncoder()
    le_movie = preprocessing.LabelEncoder()
    df_ratings["user_id"] = le_user.fit_transform(df_ratings.user_id.values)
    df_ratings["movie_id"] = le_movie.fit_transform(df_ratings.movie_id.values)

    df_train, df_valid = model_selection.train_test_split(
        df_ratings, test_size=0.2, random_state=42
    )

    num_users = len(le_user.classes_)
    num_items = len(le_movie.classes_)
    num_factors = 50

    model = MatrixFactorization(num_users, num_items, num_factors)
    model = model.to(device)

    batch_size = 256

    train_size = len(df_train)
    valid_size = len(df_valid)

    train_dataset = MovilensDataset(df_train)
    valid_dataset = MovilensDataset(df_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    lr = 5e-4
    wd = 1e-5
    epochs = 20

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    train_loss_log = []
    valid_loss_log = []

    for epoch in range(epochs):
        model.train()
        train_running_loss = 0.0
        for batch in train_dataloader:

            user_ids = batch["users"].to(device)
            item_ids = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)

            outputs = model(user_ids, item_ids)

            optimizer.zero_grad()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            valid_running_loss = 0.0
            for batch in valid_dataloader:
                user_ids = batch["users"].to(device)
                item_ids = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)

                outputs = model(user_ids, item_ids)

                loss = criterion(outputs, ratings)
                valid_running_loss += loss.item()

        train_loss = train_running_loss / train_size
        valid_loss = valid_running_loss / valid_size

        train_loss_log.append(train_loss)
        valid_loss_log.append(valid_loss)

        print(f"[epoch {epoch+1}] train loss: {train_loss:.5f}   valid loss: {valid_loss:.5f}")
