import cppimport
import numpy as np
import sys
from torch.utils.data import Dataset

from config import opt

sys.path.append('cppcode')
pbd = cppimport.imp(opt.pbd_module)

class Data(Dataset):
    def __init__(self, features,
                 num_item, num_ng=0, is_training=None):
        """features=train_data,num_item=item_num,train_mat，スパース行列，num_ng,トレーニングフェーズではデフォルトで4、つまり1つの評価されたデータに対して4つの負のサンプルを抽出します。
        """
        super(Data, self).__init__()
        """ トレーニング時にのみラベルが有用であることに注意してください。そのため、ng_sample()関数でラベルを追加します。
        """
        self.features = features  # トレーニングデータ、pd.seriesクラス
        self.num_item = num_item    # トレーニングおよびテストデータ内のitemIDの最大値
        self.num_ng = num_ng  # トレーニングフェーズでの負のサンプリング数
        self.is_training = is_training

    def ng_sample(self):  # C言語モジュールを使用して負のサンプリングを行います
        assert self.is_training, 'テスト時にはサンプリングは不要です'
        self.features_fill = []  # 各正のサンプルに対して4つの負のサンプルをマッチングします

        if opt.high_rating_train:
            # 評価値が5のものをpositiveサンプルとする
            positive_mask = np.array(self.features['rating']) == 5
            user_positive = np.array(self.features['user'])[positive_mask]
            item_positive = np.array(self.features['item'])[positive_mask]
            time_positive = np.array(self.features['timestamp'])[positive_mask]
            split_idx_positive = np.array(self.features['split_idx'])[positive_mask]
            # print('user_positive:', user_positive.shape)
            # print(user_positive)
            # print('item_positive:', item_positive.shape)
            # print(item_positive)

            user_positive = user_positive.repeat(self.num_ng)  # userIDをk回繰り返して、負のサンプルリストのサイズに一致させます
            user_positive = user_positive.reshape((user_positive.shape[0], 1))
            item_positive = item_positive.repeat(self.num_ng)  # 正のサンプルアイテムに同じ操作を行います
            item_positive = item_positive.reshape((item_positive.shape[0], 1))
            time_positive = time_positive.repeat(self.num_ng)    # タイムスタンプに同じ操作を行います
            time_positive = time_positive.reshape((time_positive.shape[0], 1))
            split_idx_positive = split_idx_positive.repeat(self.num_ng)  # タイムスタンプに同じ操作を行います
            split_idx_positive = split_idx_positive.reshape((split_idx_positive.shape[0], 1))


            # 評価値が5ではないものをnegativeサンプルとする
            negative_mask = np.array(self.features['rating']) != 5
            user_negative = np.array(self.features['user'])[negative_mask]
            item_negative = np.array(self.features['item'])[negative_mask]

            # print('user_negative:', user_negative.shape)
            # print(user_negative)
            # print('item_negative:', item_negative.shape)
            # print(item_negative)
            item_negative = item_negative.repeat(self.num_ng)
            item_negative = item_negative.reshape((item_negative.shape[0], 1))


            # 正のサンプルと負のサンプルの数を一致させる
            min_length = min(item_positive.shape[0], item_negative.shape[0])
            user_positive = user_positive[:min_length]
            item_positive = item_positive[:min_length]
            time_positive = time_positive[:min_length]
            split_idx_positive = split_idx_positive[:min_length]
            item_negative = item_negative[:min_length]

            # print('user_positive:', user_positive.shape)

            features_np = np.concatenate(
                (user_positive,
                item_positive,
                item_negative,
                time_positive,
                split_idx_positive),
                axis=1)  # 正のサンプルと負のサンプルを結合します
            # print('features_np:', features_np.shape)
        else:
            user_positive = np.array(self.features['user'])

            item_negative = pbd.negtive_sample(
                user_positive, np.array(
                    self.num_ng))  # 各インタラクションに対してC言語でk個の負のサンプルを抽出します
            item_negative = item_negative.reshape(
                (item_negative.shape[0], 1))  # 列ベクトルに変換します

            user_positive = np.array(
                self.features['user']).repeat(
                self.num_ng)  # userIDをk回繰り返して、負のサンプルリストのサイズに一致させます
            user_positive = user_positive.reshape((user_positive.shape[0], 1))
            item_positive = np.array(
                self.features['item']).repeat(
                self.num_ng)  # 正のサンプルアイテムに同じ操作を行います
            item_positive = item_positive.reshape((item_positive.shape[0], 1))
            time_positive = np.array(
                self.features['timestamp']).repeat(
                self.num_ng)    # タイムスタンプに同じ操作を行います
            time_positive = time_positive.reshape((time_positive.shape[0], 1))
            split_idx = np.array(
                self.features['split_idx']).repeat(
                self.num_ng)  # タイムスタンプに同じ操作を行います
            split_idx = split_idx.reshape((split_idx.shape[0], 1))


            features_np = np.concatenate(
                (user_positive,
                item_positive,
                item_negative,
                time_positive,
                split_idx),
                axis=1)  # 正のサンプルと負のサンプルを結合します

        self.features_fill = features_np.tolist()

    def __len__(self):
        if opt.high_rating_train:
            positive_mask = np.array(self.features['rating']) == 5
            item_positive = np.array(self.features['item'])[positive_mask]
            negative_mask = np.array(self.features['rating']) != 5
            item_negative = np.array(self.features['item'])[negative_mask]
            min_length = min(item_positive.shape[0], item_negative.shape[0])
            return min_length * self.num_ng
        else:
            return self.num_ng * len(self.features) if \
                self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        timestamp = features[idx][3]
        PDA_popularity = features[idx][4]
        return user, item_i, item_j, timestamp, PDA_popularity
