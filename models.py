import os
import sys
import time

import cppimport
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import opt
from typing import Tuple

sys.path.append("cppcode")


# base class for backbones
class BaseBackbone(nn.Module):
    def __init__(self, user_num, item_num, embed_dim):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim

        self.embed_user = None
        self.embed_item = None

    def get_optimizer_params(self) -> Tuple[dict]:
        """
        Get the parameters for the optimizer.

        Returns:
        List[Dict]: List of dictionaries containing parameters for the optimizer.
        """
        return [
            {"params": self.embed_user.weight, "lr": opt.lr},
            {"params": self.embed_item.weight, "lr": opt.lr},
        ]

    def get_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings for users and items.

        Returns:
        Tuple[Tensor, Tensor]: The user and item embeddings.
        """
        raise NotImplementedError

    def get_user_item_embeddings(self, user, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings for the given user and item indices.

        Parameters:
        user (Tensor): Tensor containing user indices.
        item (Tensor): Tensor containing item indices.

        Returns:
        Tuple[Tensor, Tensor]: The user and item embeddings.
        """
        raise NotImplementedError

    def reg_loss(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the regularization loss for the given user and items.

        Parameters:
        user (Tensor): Tensor containing user indices.
        item_i (Tensor): Tensor containing positive item indices.
        item_j (Tensor): Tensor containing negative item indices.

        Returns:
        Tensor: The regularization loss.
        """
        raise NotImplementedError

    def predict_matching(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the predictions for the given user and items.

        Parameters:
        user (Tensor): Tensor containing user indices.
        item_i (Tensor): Tensor containing positive item indices.
        item_j (Tensor): Tensor containing negative item indices.

        Returns:
        Tuple[Tensor, Tensor]: The predictions for matching between users and the positive (and negative) items.
        """
        raise NotImplementedError

    def predict_all(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predictions for all items for the given user.

        Parameters:
        user (Tensor): Tensor containing user indices.
        item (Tensor): Tensor containing item indices.

        Returns:
        Tensor
        """
        raise NotImplementedError


# backbone class for LightGCN
class LightGCNBackbone(BaseBackbone):
    def __init__(self, user_num, item_num, embed_dim, num_train_interactions):
        super().__init__(user_num, item_num, embed_dim)
        self.num_train_interactions = num_train_interactions
        self.__get_graph()
        self.embed_user = nn.Embedding(user_num, embed_dim)
        self.embed_item = nn.Embedding(item_num, embed_dim)
        nn.init.normal_(self.embed_user.weight, std=0.1)
        nn.init.normal_(self.embed_item.weight, std=0.1)

    def get_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb]).cuda()
        embs = [all_emb]
        for layer in range(opt.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return torch.split(light_out, [self.user_num, self.item_num])

    def get_user_item_embeddings(self, user, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_user(user), self.embed_item(item)

    def reg_loss(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> torch.Tensor:
        # For LightGCN, perform regularization directly on the embeddings, similar to MF
        user_emb = self.embed_user.weight[user]
        item_i_emb = self.embed_item.weight[item_i]
        item_j_emb = self.embed_item.weight[item_j]
        loss = (
            user_emb.norm(2).pow(2)
            + item_i_emb.norm(2).pow(2)
            + item_j_emb.norm(2).pow(2)
        )
        reg_loss = 0.5 * loss / float(len(user))
        return reg_loss

    def predict_matching(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb, item_i_emb = self.get_user_item_embeddings(user, item_i)
        prediction_matching_i = (user_emb * item_i_emb).sum(dim=-1)
        user_emb, item_j_emb = self.get_user_item_embeddings(user, item_j)
        prediction_matching_j = (user_emb * item_j_emb).sum(dim=-1)
        return prediction_matching_i, prediction_matching_j

    def predict_all(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        embed_user, embed_item = self.get_emb()
        return torch.matmul(embed_user[user], embed_item.T)

    def __use_dense(self) -> bool:
        density = self.num_train_interactions / (self.user_num * self.item_num)
        if self.user_num > 15000 or self.item_num > 15000:
            return False
        elif density < 0.0003:
            return False
        return True

    def __get_graph(self) -> None:
        """
        Create and load the GCN graph for the model.
        """
        # Create GCN graph
        if os.path.exists(opt.graph_data_path) and os.path.exists(opt.graph_index_path):
            graph_data = np.load(opt.graph_data_path)
            graph_index = np.load(opt.graph_index_path)
            graph_data = torch.from_numpy(graph_data)
            graph_index = torch.from_numpy(graph_index)
            graph_size = torch.Size(
                [self.user_num + self.item_num, self.user_num + self.item_num]
            )
            self.Graph = torch.sparse.FloatTensor(graph_index, graph_data, graph_size)
            self.Graph = self.Graph.coalesce().cuda()
        else:
            train_data = pd.read_csv(opt.train_data, sep="\t")
            trainUser = train_data["user"].values
            trainItem = train_data["item"].values
            user_dim = torch.LongTensor(trainUser)
            item_dim = torch.LongTensor(trainItem)
            first_sub = torch.stack([user_dim, item_dim + self.user_num])
            second_sub = torch.stack([item_dim + self.user_num, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(
                index,
                data,
                torch.Size(
                    [self.user_num + self.item_num, self.user_num + self.item_num]
                ),
            )

            if self.__use_dense():
                # For small datasets, use dense representation
                dense = self.Graph.to_dense()
                D = torch.sum(dense, dim=1).float()
                D[D == 0.0] = 1.0
                D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
                dense = dense / D_sqrt
                dense = dense / D_sqrt.t()
                index = dense.nonzero()
                data = dense[dense >= 1e-9]
                self.Graph = torch.sparse.FloatTensor(
                    index.t(),
                    data,
                    torch.Size(
                        [self.user_num + self.item_num, self.user_num + self.item_num]
                    ),
                )
            else:
                # For large datasets, use sparse representation
                deg = torch.sparse.sum(self.Graph, dim=1).to_dense()
                deg_inv_sqrt = torch.pow(deg, -0.5)
                deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
                row, col = self.Graph._indices()
                values = self.Graph._values()
                values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
                self.Graph = torch.sparse.FloatTensor(
                    self.Graph._indices(), values, self.Graph.size()
                )
                self.Graph = self.Graph.coalesce()

            np.save(opt.graph_index_path, self.Graph._indices().cpu().numpy())
            np.save(opt.graph_data_path, self.Graph._values().cpu().numpy())
            self.Graph = self.Graph.coalesce().cuda()

            # print(self.Graph, self.Graph.shape)


# backbone class for MF
class MFBackbone(BaseBackbone):
    def __init__(self, user_num, item_num, embed_dim):
        super().__init__(user_num, item_num, embed_dim)
        self.embed_user = nn.Embedding(user_num, embed_dim)
        self.embed_item = nn.Embedding(item_num, embed_dim)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def get_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_user.weight, self.embed_item.weight

    def get_user_item_embeddings(self, user, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_user(user), self.embed_item(item)

    def reg_loss(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.embed_user.weight[user]
        item_i_emb = self.embed_item.weight[item_i]
        item_j_emb = self.embed_item.weight[item_j]
        loss = (
            user_emb.norm(2).pow(2)
            + item_i_emb.norm(2).pow(2)
            + item_j_emb.norm(2).pow(2)
        )
        reg_loss = 0.5 * loss / float(len(user))
        return reg_loss

    def predict_matching(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb, item_i_emb = self.get_user_item_embeddings(user, item_i)
        prediction_matching_i = (user_emb * item_i_emb).sum(dim=-1)
        user_emb, item_j_emb = self.get_user_item_embeddings(user, item_j)
        prediction_matching_j = (user_emb * item_j_emb).sum(dim=-1)
        return prediction_matching_i, prediction_matching_j

    def predict_all(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        embed_user, embed_item = self.get_emb()
        return torch.matmul(embed_user[user], embed_item.T)


# backbone class for FM
class FMBackbone(BaseBackbone):
    def __init__(self, user_num, item_num, embed_dim):
        super().__init__(user_num, item_num, embed_dim)
        self.embed_user = nn.Embedding(user_num, embed_dim)
        self.embed_item = nn.Embedding(item_num, embed_dim)
        self.bias_user = nn.Embedding(user_num, 1)
        self.bias_item = nn.Embedding(item_num, 1)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.constant_(self.bias_user.weight, 0.0)
        nn.init.constant_(self.bias_item.weight, 0.0)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def get_optimizer_params(self) -> Tuple[dict]:
        return [
            {"params": self.embed_user.weight, "lr": opt.lr},
            {"params": self.embed_item.weight, "lr": opt.lr},
            {"params": self.bias_user.weight, "lr": opt.lr},
            {"params": self.bias_item.weight, "lr": opt.lr},
            {"params": self.global_bias, "lr": opt.lr},
        ]

    def get_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_user.weight, self.embed_item.weight

    def get_user_item_embeddings(self, user, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_user(user), self.embed_item(item)

    def reg_loss(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.embed_user.weight[user]
        item_i_emb = self.embed_item.weight[item_i]
        item_j_emb = self.embed_item.weight[item_j]
        loss = (
            user_emb.norm(2).pow(2)
            + item_i_emb.norm(2).pow(2)
            + item_j_emb.norm(2).pow(2)
            + self.bias_user.weight[user].norm(2).pow(2)
            + self.bias_item.weight[item_i].norm(2).pow(2)
            + self.bias_item.weight[item_j].norm(2).pow(2)
            + self.global_bias.norm(2).pow(2)
        )
        reg_loss = 0.5 * loss / float(len(user))
        return reg_loss

    def predict_matching(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb, item_i_emb = self.get_user_item_embeddings(user, item_i)
        prediction_matching_i = (
            self.global_bias
            + self.bias_user(user).squeeze()
            + self.bias_item(item_i).squeeze()
            + (user_emb * item_i_emb).sum(dim=-1)
        )
        user_emb, item_j_emb = self.get_user_item_embeddings(user, item_j)
        prediction_matching_j = (
            self.global_bias
            + self.bias_user(user).squeeze()
            + self.bias_item(item_j).squeeze()
            + (user_emb * item_j_emb).sum(dim=-1)
        )
        return prediction_matching_i, prediction_matching_j

    def predict_all(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_emb = self.embed_user(user)  # (B, d)
        item_emb = self.embed_item(item)  # (N, d)
        user_bias = self.bias_user(user).squeeze()  # (B,)
        item_bias = self.bias_item(item).squeeze()  # (N,)
        dot = torch.matmul(user_emb, item_emb.t())  # (B, N)
        return self.global_bias + user_bias.unsqueeze(1) + item_bias.unsqueeze(0) + dot

    def __get_biases(self, user, item):
        return self.bias_user(user).squeeze(), self.bias_item(item).squeeze()


# backbone class for NCF
class NCFBackbone(BaseBackbone):
    def __init__(self, user_num, item_num, embed_dim, fc_dims):
        super().__init__(user_num, item_num, embed_dim)
        self.embedding_user = nn.Embedding(user_num, embed_dim)
        self.embedding_item = nn.Embedding(item_num, embed_dim)
        fc_input = 2 * embed_dim
        self.fc_layers = nn.ModuleList()
        for out_dim in fc_dims:
            self.fc_layers.append(nn.Linear(fc_input, out_dim))
            fc_input = out_dim
        self.affine_output = nn.Linear(fc_input, 1)
        self.logistic = nn.Sigmoid()

    def get_optimizer_params(self) -> Tuple[dict]:
        return [
            {"params": self.embedding_user.weight, "lr": opt.lr},
            {"params": self.embedding_item.weight, "lr": opt.lr},
            {"params": self.fc_layers.parameters(), "lr": opt.lr},
            {"params": self.affine_output.parameters(), "lr": opt.lr},
        ]

    def get_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embedding_user.weight, self.embedding_item.weight

    def get_user_item_embeddings(self, user, item):
        return self.embedding_user(user), self.embedding_item(item)

    def reg_loss(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.embedding_user.weight[user]
        item_i_emb = self.embedding_item.weight[item_i]
        item_j_emb = self.embedding_item.weight[item_j]
        loss = (
            user_emb.norm(2).pow(2)
            + item_i_emb.norm(2).pow(2)
            + item_j_emb.norm(2).pow(2)
        )
        reg_loss = 0.5 * loss / float(len(user))
        return reg_loss

    def predict_matching(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb, item_i_emb = self.get_user_item_embeddings(user, item_i)
        x = torch.cat([user_emb, item_i_emb], dim=1)
        for idx, layer in enumerate(self.fc_layers):
            x = F.relu(layer(x))
        prediction_matching_i = self.logistic(self.affine_output(x))
        user_emb, item_j_emb = self.get_user_item_embeddings(user, item_j)
        x = torch.cat([user_emb, item_j_emb], dim=1)
        for idx, layer in enumerate(self.fc_layers):
            x = F.relu(layer(x))
        prediction_matching_j = self.logistic(self.affine_output(x))
        return prediction_matching_i, prediction_matching_j

    def predict_all(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        For NCF, compute prediction scores for all candidate items using an MLP.
        """
        user_emb = self.embedding_user(user)  # (B, d)
        item_emb = self.embedding_item(item)  # (N, d)

        self.eval()
        batch_size = 126
        with torch.no_grad():
            user_emb = self.embedding_user(user)  # (B, d)
            B = user_emb.size(0)
            scores = []
            N = item.size(0)
            # Process items in batches to avoid memory issues
            for i in range(0, N, batch_size):
                batch_items = item[
                    i : i + batch_size
                ]  # Get indices for the batch size
                batch_item_emb = self.embedding_item(batch_items)  # (batch_size, d)
                # Expand embeddings to create all user-item pairs for the batch.
                user_emb_exp = user_emb.unsqueeze(1).expand(
                    B, batch_item_emb.size(0), -1
                )  # (B, batch_size, d)
                item_emb_exp = batch_item_emb.unsqueeze(0).expand(
                    B, batch_item_emb.size(0), -1
                )  # (B, batch_size, d)
                x = torch.cat(
                    [user_emb_exp, item_emb_exp], dim=2
                )  # (B, batch_size, 2*d)
                x = x.contiguous().view(-1, x.size(2))  # (B*batch_size, 2*d)
                for layer in self.fc_layers:
                    x = F.relu(layer(x))
                batch_scores = self.logistic(self.affine_output(x))
                scores.append(batch_scores.view(B, -1))
            return torch.cat(scores, dim=1)


class Models(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, min_time, max_time, num_train_interactions
    ):
        super(Models, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.n_users = user_num
        self.m_items = item_num
        self.embed_size = factor_num
        self.num_train_interactions = num_train_interactions

        if opt.backbone == "LightGCN":
            self.backbone = LightGCNBackbone(
                user_num, item_num, factor_num, num_train_interactions
            )
        elif opt.backbone == "MF":
            self.backbone = MFBackbone(user_num, item_num, factor_num)
        elif opt.backbone == "FM":
            self.backbone = FMBackbone(user_num, item_num, factor_num)
        elif opt.backbone == "NCF":
            # opt.n_layersはリストで指定
            self.backbone = NCFBackbone(
                user_num,
                item_num,
                opt.ncf_embedding_dim if hasattr(opt, "ncf_embedding_dim") else 5,
                list(opt.n_layers),
            )
        else:
            raise ValueError("Invalid backbone specified")

        # Parameter settings
        if opt.use_q:
            self.q = nn.Parameter(torch.ones(item_num) * opt.q)
        if opt.use_a:
            self.a = nn.Parameter(torch.ones(item_num) * opt.a)
        if opt.use_b:
            self.b = nn.Parameter(torch.ones(item_num) * opt.b)

        self.tau = torch.ones(item_num) * opt.tau  # 100000000

        self.pbd_import()

        if opt.use_cti:
            self.pbd.load_popularity(self.tau.cpu().detach().numpy())
        else:
            self.pbd.load_popularity_count_review()

        if opt.useRatings:
            if opt.normalizedRatings:
                print("loading norm ratings...", flush=True)
                self.pbd.load_ratings_norm()
            else:
                print("loading ratings...", flush=True)
                self.pbd.load_ratings()

        self.max_time = max_time
        self.min_time = min_time
        PDA_popularity = pd.read_csv(opt.PDA_popularity_path, sep="\t")
        self.PDA_array = PDA_popularity.values.reshape(-1)
        self.DICE_pop = np.load(opt.DICE_popularity_path)

        print("model init done")

    def pbd_import(self):
        """
        Import the appropriate PBD module based on the dataset specified in the options.
        """
        print("model loading...", opt.dataset, flush=True)
        self.pbd = cppimport.imp(opt.pbd_module)
        print("model loaded")

    def get_initializer(self):
        # まずバックボーンのパラメータを取得
        params = self.backbone.get_optimizer_params()
        # 追加パラメータは Models 側に定義されているのでここで結合する
        if opt.use_q:
            params.append(
                {"params": self.q, "lr": opt.lr_q, "weight_decay": opt.lamb_q}
            )
        if opt.use_a:
            params.append(
                {"params": self.a, "lr": opt.lr_a, "weight_decay": opt.lamb_a}
            )
        if opt.use_b:
            params.append(
                {"params": self.b, "lr": opt.lr_b, "weight_decay": opt.lamb_b}
            )
        return torch.optim.Adam(params)

    def get_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings for users and items.

        Returns:
        Tuple[Tensor, Tensor]: The user and item embeddings.
        """
        return self.backbone.get_emb()

    def reg_loss(self, user, item_i, item_j) -> torch.Tensor:
        """
        Calculate the regularization loss for the given user and items.

        Parameters:
        user (Tensor): Tensor containing user indices.
        item_i (Tensor): Tensor containing positive item indices.
        item_j (Tensor): Tensor containing negative item indices.

        Returns:
        Tensor: The regularization loss.
        """
        return self.backbone.reg_loss(user, item_i, item_j)

    def forward(self, user, item_i, item_j, timestamp, split_idx):
        """
        Forward pass for the model.

        Parameters:
        user (Tensor): Tensor containing user indices.
        item_i (Tensor): Tensor containing positive item indices.
        item_j (Tensor): Tensor containing negative item indices.
        timestamp (Tensor): Tensor containing timestamps.
        split_idx (Tensor): Tensor containing split indices.

        Returns:
        Depending on the method specified in opt, returns different values such as predictions, losses, and regularization loss.
        """
        embed_user, embed_item = self.get_emb()

        # Check if item indices are within valid range
        if torch.any(item_i >= self.m_items):
            raise IndexError(f"item_i is out of range: {item_i}")
        if torch.any(item_j >= self.m_items):
            raise IndexError(f"item_j is out of range: {item_j}")

        user_embedding = embed_user[user]
        item_i_embedding = embed_item[item_i]
        item_j_embedding = embed_item[item_j]
        reg_loss = self.reg_loss(user, item_i, item_j)

        if opt.method == "base":  # Calculate popularity directly without computing it
            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            return prediction_matching_i, prediction_matching_j, reg_loss
        elif opt.method == "ratingBase":
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            ari_i = torch.from_numpy(self.pbd.ari(item_i_np, timestamp)).cuda().float()
            ari_j = torch.from_numpy(self.pbd.ari(item_j_np, timestamp)).cuda().float()
            self.popularity_i = ari_i
            self.popularity_j = ari_j
            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(ari_i)
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(ari_j)
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method == "IPS":
            item_i = item_i.cpu().numpy().astype(int)
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            IPS_c = (
                torch.from_numpy(
                    np.minimum(1 / self.DICE_pop[item_i], opt.IPS_lambda)
                ).cuda()
                / opt.IPS_lambda
            )
            ips_loss = (
                -1 * (IPS_c * (prediction_i - prediction_j).sigmoid().log()).sum()
                + opt.lamb * reg_loss
            )
            return ips_loss
        elif opt.method == "DICE":
            if opt.backbone == "NCF":
                # For NCF, retrieve the actual embedding dimension
                actual_embed_dim = self.backbone.embed_dim
                if actual_embed_dim < 4:  # At least 4 dimensions are required (for 2x2 splitting)
                    raise ValueError(
                        f"NCF embedding dimension ({actual_embed_dim}) is too small for DICE method. Need at least 4 dimensions."
                    )
                DICE_size = int(actual_embed_dim / 2)
            else:
                DICE_size = int(self.embed_size / 2)

            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            # Loss for click prediction
            loss_click = -(prediction_i - prediction_j).sigmoid().log().sum()

            if opt.backbone in ["LightGCN", "MF", "FM"]:
                user_embedding_1 = user_embedding[:, 0:DICE_size]
                item_i_embedding_1 = item_i_embedding[:, 0:DICE_size]
                item_j_embedding_1 = item_j_embedding[:, 0:DICE_size]
                user_embedding_2 = user_embedding[:, DICE_size:]
                item_i_embedding_2 = item_i_embedding[:, DICE_size:]
                item_j_embedding_2 = item_j_embedding[:, DICE_size:]
            else:  # NCF and other backbones
                # For NCF, adjust sizes in case of odd embedding dimensions
                actual_size_1 = DICE_size
                actual_size_2 = user_embedding.size(1) - DICE_size  # Calculate remaining size accurately

                user_embedding_1 = user_embedding[:, 0:actual_size_1]
                item_i_embedding_1 = item_i_embedding[:, 0:actual_size_1]
                item_j_embedding_1 = item_j_embedding[:, 0:actual_size_1]
                user_embedding_2 = user_embedding[:, DICE_size:DICE_size+actual_size_1]  # Limit to the same size
                item_i_embedding_2 = item_i_embedding[:, DICE_size:DICE_size+actual_size_1]
                item_j_embedding_2 = item_j_embedding[:, DICE_size:DICE_size+actual_size_1]
            loss_discrepancy = -1 * (
                (user_embedding_1 - user_embedding_2).sum()
                + (item_i_embedding_1 - item_i_embedding_2).sum()
                + (item_j_embedding_1 - item_j_embedding_2).sum()
            )

            item_i_np = item_i.cpu().numpy().astype(int)
            item_j_np = item_j.cpu().numpy().astype(int)
            pop_relation = self.DICE_pop[item_i_np] > self.DICE_pop[item_j_np]
            user_O1 = user[pop_relation]
            user_O2 = user[~pop_relation]
            item_i_O1 = item_i[pop_relation]
            item_j_O1 = item_j[pop_relation]
            item_i_O2 = item_i[~pop_relation]
            item_j_O2 = item_j[~pop_relation]

            if opt.backbone in ["LightGCN", "MF", "FM"]:
                user_embedding_for_interest = user_embedding[:, 0:DICE_size]
                item_i_embedding_for_interest = item_i_embedding[:, 0:DICE_size]
                item_j_embedding_for_interest = item_j_embedding[:, 0:DICE_size]
            else:  # NCF and other backbones
                # Use data filtered by pop_relation
                if pop_relation.any():  # Process only if there are True elements in pop_relation
                    user_embedding_for_interest = embed_user[user_O1][:, 0:DICE_size]
                    item_i_embedding_for_interest = embed_item[item_i_O1][
                        :, 0:DICE_size
                    ]
                    item_j_embedding_for_interest = embed_item[item_j_O1][
                        :, 0:DICE_size
                    ]
                else:
                    # If there are no True elements in pop_relation, create empty tensors
                    device = user_embedding.device
                    user_embedding_for_interest = torch.empty(
                        0, DICE_size, device=device
                    )
                    item_i_embedding_for_interest = torch.empty(
                        0, DICE_size, device=device
                    )
                    item_j_embedding_for_interest = torch.empty(
                        0, DICE_size, device=device
                    )

            if user_embedding_for_interest.size(0) > 0:
                prediction_i = (
                    user_embedding_for_interest * item_i_embedding_for_interest
                ).sum(dim=-1)
                prediction_j = (
                    user_embedding_for_interest * item_j_embedding_for_interest
                ).sum(dim=-1)
                # Loss for interest prediction
                loss_interest = -(prediction_i - prediction_j).sigmoid().log().sum()
            else:
                loss_interest = torch.tensor(0.0, device=user_embedding.device)

            if opt.backbone in ["LightGCN", "MF", "FM"]:
                user_embedding_for_pop1 = user_embedding[:, DICE_size:]
                item_i_embedding_for_pop1 = item_i_embedding[:, DICE_size:]
                item_j_embedding_for_pop1 = item_j_embedding[:, DICE_size:]
            else:  # NCF and other backbones
                if pop_relation.any():
                    user_embedding_for_pop1 = embed_user[user_O1][:, DICE_size:]
                    item_i_embedding_for_pop1 = embed_item[item_i_O1][:, DICE_size:]
                    item_j_embedding_for_pop1 = embed_item[item_j_O1][:, DICE_size:]
                else:
                    device = user_embedding.device
                    user_embedding_for_pop1 = torch.empty(0, DICE_size, device=device)
                    item_i_embedding_for_pop1 = torch.empty(0, DICE_size, device=device)
                    item_j_embedding_for_pop1 = torch.empty(0, DICE_size, device=device)

            if user_embedding_for_pop1.size(0) > 0:
                prediction_i = (
                    user_embedding_for_pop1 * item_i_embedding_for_pop1
                ).sum(dim=-1)
                prediction_j = (
                    user_embedding_for_pop1 * item_j_embedding_for_pop1
                ).sum(dim=-1)
                # Loss for p1
                loss_popularity_1 = -(prediction_j - prediction_i).sigmoid().log().sum()
            else:
                loss_popularity_1 = torch.tensor(0.0, device=user_embedding.device)

            if opt.backbone in ["LightGCN", "MF", "FM"]:
                user_embedding_for_pop2 = user_embedding[:, DICE_size:]
                item_i_embedding_for_pop2 = item_i_embedding[:, DICE_size:]
                item_j_embedding_for_pop2 = item_j_embedding[:, DICE_size:]
            else:  # NCF and other backbones
                if (~pop_relation).any():
                    user_embedding_for_pop2 = embed_user[user_O2][:, DICE_size:]
                    item_i_embedding_for_pop2 = embed_item[item_i_O2][:, DICE_size:]
                    item_j_embedding_for_pop2 = embed_item[item_j_O2][:, DICE_size:]
                else:
                    device = user_embedding.device
                    user_embedding_for_pop2 = torch.empty(0, DICE_size, device=device)
                    item_i_embedding_for_pop2 = torch.empty(0, DICE_size, device=device)
                    item_j_embedding_for_pop2 = torch.empty(0, DICE_size, device=device)

            if user_embedding_for_pop2.size(0) > 0:
                prediction_i = (
                    user_embedding_for_pop2 * item_i_embedding_for_pop2
                ).sum(dim=-1)
                prediction_j = (
                    user_embedding_for_pop2 * item_j_embedding_for_pop2
                ).sum(dim=-1)
                # p2の損失
                loss_popularity_2 = -(prediction_i - prediction_j).sigmoid().log().sum()
            else:
                loss_popularity_2 = torch.tensor(0.0, device=user_embedding.device)
            # print(loss_click, loss_interest, loss_popularity_1, loss_popularity_2)

            # dice_loss = loss_click + 0.1*(loss_interest + loss_popularity_1 + loss_popularity_2)
            return (
                loss_click,
                loss_interest,
                loss_popularity_1,
                loss_popularity_2,
                loss_discrepancy,
                reg_loss,
            )
        elif opt.method == "PDA" or opt.method == "PD":
            item_i = item_i.cpu().numpy()
            item_j = item_j.cpu().numpy()
            PDA_idx_i = (item_i * 10 + split_idx).astype(int)
            PDA_popularity_i = self.PDA_array[PDA_idx_i]
            PDA_popularity_i = torch.from_numpy(PDA_popularity_i).cuda()
            PDA_idx_j = (item_j * 10 + split_idx).astype(int)
            PDA_popularity_j = self.PDA_array[PDA_idx_j]
            PDA_popularity_j = torch.from_numpy(PDA_popularity_j).cuda()
            prediction_i = (
                F.elu((user_embedding * item_i_embedding).sum(dim=-1)) + 1
            ) * (PDA_popularity_i**opt.PDA_gamma)
            prediction_j = (
                F.elu((user_embedding * item_j_embedding).sum(dim=-1)) + 1
            ) * (PDA_popularity_j**opt.PDA_gamma)
            # print(PDA_popularity_i ** opt.PDA_gamma, PDA_popularity_j ** opt.PDA_gamma)
            return prediction_i, prediction_j, reg_loss
        elif opt.method == "TIDE-noc":
            self.popularity_i = F.softplus(self.q[item_i])
            self.popularity_j = F.softplus(self.q[item_j])
            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method == "TIDE-noq":
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)
            ).cuda()
            self.popularity_i = F.softplus(self.b[item_i]) * popularity_i_s

            popularity_j_s = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)
            ).cuda()
            self.popularity_j = F.softplus(self.b[item_j]) * popularity_j_s

            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method in [
            "SbC",
            "SbC-e",
            "SbC-C",
            "SbC-counts",
            "SbC-counts-e",
            "SbC-counts-C",
        ]:
            # print(opt.method)
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            if torch.isnan(item_i).any():
                raise ValueError("NaN detected in item_i")
            if torch.isnan(item_j).any():
                raise ValueError("NaN detected in item_j")

            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            ari_i = torch.from_numpy(self.pbd.ari(item_i_np, timestamp)).cuda().float()

            item_i = item_i.cuda()
            # for idx, item in enumerate(item_i):
            #     self.a[item] = ari_i[idx]   # item_iのARIをqに代入
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)
            ).cuda()
            self.popularity_i = ari_i + F.softplus(self.b[item_i]) * popularity_i_s

            ari_j = torch.from_numpy(self.pbd.ari(item_j_np, timestamp)).cuda().float()

            item_j = item_j.cuda()
            popularity_j_s = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)
            ).cuda()
            self.popularity_j = ari_j + F.softplus(self.b[item_j]) * popularity_j_s
            
            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method in [
            "aSbC",
            "aSbC-e",
            "aSbC-C",
            "aSbC-S",
            "aSbC-fixa",
            "aSbC-fixa-e",
            "aSbC-fixa-C",
            "aSbC-fixa-S",
            "aSbC-counts",
            "aSbC-counts-e",
            "aSbC-counts-C",
            "aSbC-counts-S",
        ]:
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            ari_i = torch.from_numpy(self.pbd.ari(item_i_np, timestamp)).cuda()
            """
            item_iはselectingされたitemのIDのリスト(tensor)
            item_i_npはitem_iのNumPy配列
            ari_iはitem_iの対応する(timestampの時の)ARIのリスト(tensor)
            self.a[idx]はitemのIDがidxのitemのqの値
            """
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)
            ).cuda()
            self.popularity_i = (
                F.softplus(self.a[item_i]) * ari_i
                + F.softplus(self.b[item_i]) * popularity_i_s
            )

            ari_j = torch.from_numpy(self.pbd.ari(item_j_np, timestamp)).cuda()
            self.popularity_j = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)
            ).cuda()
            self.popularity_j = (
                F.softplus(self.a[item_j]) * ari_j
                + F.softplus(self.b[item_j]) * self.popularity_j
            )

            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method == "aSbC-noc":
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            ari_i = torch.from_numpy(self.pbd.ari(item_i_np, timestamp)).cuda()
            self.popularity_i = F.softplus(self.a[item_i]) * ari_i

            ari_j = torch.from_numpy(self.pbd.ari(item_j_np, timestamp)).cuda()
            self.popularity_j = F.softplus(self.a[item_j]) * ari_j

            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method in [
            "qaSbC",
            "qaSbC-e",
            "qaSbC-C",
            "qaSbC-S",
            "qaSbC-SC",
            "qaSbC-counts",
            "qaSbC-counts-e",
            "qaSbC-counts-C",
            "qaSbC-counts-S",
            "qaSbC-counts-SC",
        ]:
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            ari_i = torch.from_numpy(self.pbd.ari(item_i_np, timestamp)).cuda()
            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)
            ).cuda()
            self.popularity_i = (
                F.softplus(self.q[item_i])
                + F.softplus(self.a[item_i]) * ari_i
                + F.softplus(self.b[item_i]) * popularity_i_s
            )

            ari_j = torch.from_numpy(self.pbd.ari(item_j_np, timestamp)).cuda()
            self.popularity_j = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)
            ).cuda()
            self.popularity_j = (
                F.softplus(self.q[item_j])
                + F.softplus(self.a[item_j]) * ari_j
                + F.softplus(self.b[item_j]) * self.popularity_j
            )

            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method in [
            "qaSbC-noc",
            "qaSbC-noc-e",
            "qaSbC-noc-C",
            "qaSbC-noc-S",
            "qaSbC-noc-SC",
        ]:
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            # t0 = time.time()
            ari_i = torch.from_numpy(self.pbd.ari(item_i_np, timestamp)).cuda()
            self.popularity_i = (
                F.softplus(self.q[item_i]) + F.softplus(self.a[item_i]) * ari_i
            )

            ari_j = torch.from_numpy(self.pbd.ari(item_j_np, timestamp)).cuda()
            self.popularity_j = (
                F.softplus(self.q[item_j]) + F.softplus(self.a[item_j]) * ari_j
            )

            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        elif opt.method in [
            "TIDE",
            "TIDE-e",
            "TIDE-C",
            "TIDE-fixq",
        ]:
            # self.pbd.load_popularity(self.tau.cpu().detach().numpy())

            # t0 = time.time()
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()

            popularity_i_s = torch.from_numpy(
                self.pbd.popularity(item_i_np, timestamp)
            ).cuda()
            self.popularity_i = (
                F.softplus(self.q[item_i]) + F.softplus(self.b[item_i]) * popularity_i_s
            )

            self.popularity_j = torch.from_numpy(
                self.pbd.popularity(item_j_np, timestamp)
            ).cuda()
            self.popularity_j = (
                F.softplus(self.q[item_j])
                + F.softplus(self.b[item_j]) * self.popularity_j
            )

            prediction_matching_i, prediction_matching_j = (
                self.backbone.predict_matching(user, item_i, item_j)
            )
            self.prediction_i = F.softplus(prediction_matching_i) * torch.tanh(
                self.popularity_i
            )
            self.prediction_j = F.softplus(prediction_matching_j) * torch.tanh(
                self.popularity_j
            )
            # print(torch.tanh(self.popularity_i), torch.tanh(self.popularity_j))
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
        else:
            raise ValueError("Invalid method")
