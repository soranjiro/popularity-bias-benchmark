import cppimport
import numpy as np
import torch
from config import opt
import torch.nn.functional as F
import sys
import time
import pandas as pd

import util
sys.path.append('cppcode')


# import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, model, user_num, item_num, max_time):
        self.model = model
        self.user_num = user_num
        self.item_num = item_num
        self.max_time = max_time
        self.pbd_import()
        # Record the best test results for the two datasets separately
        self.best_perf = {'recall': np.zeros((1,)), 'map': np.zeros((1,)),
                'ndcg': np.zeros((1,)), 'hit_rate': np.zeros((1,)),
                'averageRating': np.zeros((1,)),
                'recall@k': np.zeros((1,)), 'map@k': np.zeros((1,)),
                'ndcg@k': np.zeros((1,)), 'hit_rate@k': np.zeros((1,)),
                'best_epoch': np.zeros((1,))}
        # Load interaction records for each user in the training and test sets for use during the testing phase
        self.train_items = {}
        with open(opt.train_list) as f_train:
            for l in f_train.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split('\t')]
                uid, train_items_tmp = items[0], items[2:]
                self.train_items[uid] = train_items_tmp
        self.test_set = {}
        with open(opt.test_list) as f_test:
            for l in f_test.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split('\t')]
                uid, test_items = items[0], items[1:]
                self.test_set[uid] = test_items
        self.test_set_h = {}
        with open(opt.test_list_h) as f_test_h:
            for l in f_test_h.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split('\t')]
                uid, test_items = items[0], items[1:]
                self.test_set_h[uid] = test_items
        self.test_set_rating = {}
        with open(opt.test_list_rating) as f_test:
            for l in f_test.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [float(i) for i in l.split('\t')]
                uid, test_ratings = items[0], items[1:]
                self.test_set_rating[uid] = test_ratings

    def pbd_import(self):
        print(f"Evaluate loading {opt.pbd_module} for dataset {opt.dataset}...", flush=True)
        self.pbd = cppimport.imp(opt.pbd_module)
        print("Evaluate loaded.")

    def elu(self, x):
        return np.where(x > 0, x, np.exp(x) - 1) + 1

    def get_dcg(self, y_pred, y_true, k):
        dcg = np.zeros(len(y_true))
        for udx in range(len(y_true)):
            for i in range(1, k + 1):
                # Calculate DCG for each user
                if i - 1 >= len(y_pred[udx]):
                    break
                if y_pred[udx][i - 1] in y_true[udx]:
                    dcg[udx] += 1 / np.log2(i + 1)
            # y_pred_udx = np.array(y_pred[udx][:k])
            # y_true_udx = np.array(y_true[udx])
            # hits = np.isin(y_pred_udx, y_true_udx)
            # dcg[udx] = np.sum(hits / np.log2(np.arange(1, len(hits) + 1) + 1))
        return dcg

    def get_idcg(self, test_items, k):
        idcg = np.zeros(len(test_items))
        for udx in range(len(test_items)):
            limit = min(len(test_items[udx]), k)
            for i in range(1, limit + 1):
                idcg[udx] += 1 / np.log2(i + 1)
            # idcg[udx] = np.sum(1 / np.log2(np.arange(1, limit + 1) + 1))
        return idcg

    def get_ndcg(self, sorted_pred, test_items, k):
        dcg = self.get_dcg(sorted_pred, test_items, k)
        idcg = self.get_idcg(test_items, k)
        # Suppress warnings for division by zero and invalid operations
        with np.errstate(divide='ignore', invalid='ignore'):
            ndcg = np.divide(dcg, idcg, out=np.full_like(dcg, np.nan), where=idcg!=0)
        return ndcg

    def get_map(self, sorted_pred, test_items, k):
        map_score = np.zeros(len(test_items))
        for udx in range(len(test_items)):
            if len(test_items[udx]) == 0:
                continue
            avg_precision = 0
            hit = 0
            for i in range(1, k + 1):
                if i - 1 < len(sorted_pred[udx]) and sorted_pred[udx][i - 1] in test_items[udx]:
                    hit += 1
                    avg_precision += hit / i
            map_score[udx] = avg_precision / min(len(test_items[udx]), k)
            # relevant = np.isin(sorted_pred[udx][:k], test_items[udx])
            # hits = np.cumsum(relevant)
            # precision_at_i = hits / np.arange(1, len(hits) + 1)
            # avg_precision = np.sum(precision_at_i * relevant) / min(len(test_items[udx]), k)
            # map_score[udx] = avg_precision
        return map_score

    def get_hit_rate(self, sorted_pred, test_items, k):
        hit_rate = np.zeros(len(test_items))
        for udx in range(len(test_items)):
            for i in range(k):
                if i < len(sorted_pred[udx]) and sorted_pred[udx][i] in test_items[udx]:
                    hit_rate[udx] = 1
                    break
            # hit_rate[udx] = np.any(np.isin(sorted_pred[udx][:k], test_items[udx]))
        return hit_rate

    def auc(self, predicted_list, test_item_h):
        label = np.isin(predicted_list, test_item_h)
        m = np.sum(label == 1)
        n = np.sum(label == 0)
        if m == 0:
            return np.nan  # 0
        if n == 0:
            return np.nan  # 1
        # print(label)
        # print(np.arange(len(predicted_list), 0, -1))
        # print(np.arange(len(predicted_list), 0, -1) * label)
        auc = (np.sum(np.arange(len(predicted_list), 0, -1)
               * label) - (m * (m + 1) / 2)) / m / n
        # print(auc)
        return auc

    def evaluate(
            self,
            rate_batch,
            test_items,
            test_items_h,
            top_k,
            top_at_k,
            user_batch):
        # Store recall, map, ndcg, and hit_rate for each user. Results for two types of test sets are distinguished in the last dimension.
        result = np.zeros((len(rate_batch), 9))
        pred = np.argpartition(
            rate_batch, -top_k)[:, -top_k:]   # Use np.argpartition to get the top-k items for each user
        sorted_pred = np.zeros(np.shape(pred))
        for udx in range(len(rate_batch)):
            sorted_pred[udx] = pred[udx, np.argsort(rate_batch[udx, pred[udx]])][::-1]   # Sorted recommendation list

        if opt.test_only and opt.save_recommend and top_k == 25:
            original_options = np.get_printoptions()
            np.set_printoptions(threshold=sys.maxsize)
            util.print_str(opt.recommend_path, str(sorted_pred), True, False)
            util.print_str(opt.recommend_path, "\n\n\n\n", True, False)
            np.set_printoptions(**original_options)

        for udx in range(len(rate_batch)):
            u = user_batch[udx]
            hit = 0
            rating_sum = 0
            rating_vacant = 0
            for i in pred[udx]:
                if i in test_items[udx]:
                    hit += 1
                    idx = test_items[udx].index(i)
                    if self.test_set_rating[u][idx] == -1:
                        rating_vacant += 1
                    else:
                        rating_sum += self.test_set_rating[u][idx]

            # Calculate recall
            result[udx][0] = hit / len(test_items[udx])  # recall
            # Calculate average rating
            result[udx][4] = rating_sum / (hit - rating_vacant) if (
                hit - rating_vacant) > 0 else 0  # average_rating

        # MAP
        result[:, 1] = self.get_map(sorted_pred, test_items, top_k)

        # Calculate NDCG
        result[:, 2] = self.get_ndcg(sorted_pred, test_items, top_k)

        # Calculate Hit Rate
        result[:, 3] = self.get_hit_rate(sorted_pred, test_items, top_k)

        # sorted_pred_2 = {udx: np.array(test_items[udx])[np.argsort(-rate_batch[udx, test_items[udx]])] for udx in range(len(rate_batch))}
        sorted_pred_2 = {}
        for udx in range(len(rate_batch)):
            sorted_pred_2[udx] = np.array(test_items[udx])[np.argsort(-1 * rate_batch[udx, test_items[udx]])]

        if opt.test_only and opt.save_recommend and top_at_k == 25:
            original_options = np.get_printoptions()
            np.set_printoptions(threshold=sys.maxsize)
            util.print_str(opt.recommend_path, str(sorted_pred_2), True, False)
            util.print_str(opt.recommend_path, "\n\n\n\n", True, False)
            np.set_printoptions(**original_options)

        for udx in range(len(rate_batch)):
            # if len(test_items_h[udx]) < top_at_k or len(test_items[udx]) < top_at_k:
            #     result[udx, 5] = np.nan
            #     result[udx, 6] = np.nan
            #     result[udx, 7] = np.nan
            #     result[udx, 8] = np.nan
            if len(test_items_h[udx]) - len(test_items[udx]) == 0:
                result[udx, 5] = np.nan
                result[udx, 6] = np.nan
                result[udx, 7] = np.nan
                result[udx, 8] = np.nan
            elif len(test_items_h[udx]) == 0:
                result[udx, 5] = np.nan
                result[udx, 6] = np.nan
                result[udx, 7] = np.nan
                result[udx, 8] = np.nan
            else:
                hit = 0
                for i in sorted_pred_2[udx][0:top_at_k]:
                    if i in test_items_h[udx]:
                        hit += 1
                # Calculate recall@k, map@k, ndcg@k, hit_rate@k
                result[udx][5] = hit / len(test_items_h[udx])  # recall@k
                result[udx][6] = self.get_map(sorted_pred_2, test_items_h, top_at_k)[udx]  # map@k
                result[udx][7] = self.get_ndcg(sorted_pred_2, test_items_h, top_at_k)[udx]  # ndcg@k
                result[udx][8] = self.get_hit_rate(sorted_pred_2, test_items_h, top_at_k)[udx]  # hit_rate@k

        return result

    def test(self, users_to_test, popularity_item_np):
        USR_NUM, ITEM_NUM = self.user_num, self.item_num
        BATCH_SIZE = 8192 * 4
        top_show = np.array([opt.topk])
        max_top = max(top_show)
        top_at_k = opt.top_at_k
        result = {
            'recall': np.zeros(len(top_show)),
            'map': np.zeros(len(top_show)),
            'ndcg': np.zeros(len(top_show)),
            'hit_rate': np.zeros(len(top_show)),
            'averageRating': np.zeros(len(top_show)),
            'recall@k': np.zeros(len(top_show)),
            'map@k': np.zeros(len(top_show)),
            'ndcg@k': np.zeros(len(top_show)),
            'hit_rate@k': np.zeros(len(top_show)),
        }
        u_batch_size = BATCH_SIZE  # Number of users per batch
        test_users = users_to_test  # Users included in the test set
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        if opt.debug:
            print("test_users", test_users)
            print("n_test_users", n_test_users)
            print("n_user_batchs", n_user_batchs)
            print("u_batch_size", u_batch_size)

        all_result = []
        item_batch = range(ITEM_NUM)  # All items participate in ranking
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            # beta = model.beta.detach().cpu().numpy()
            # print("beta=", beta.mean(), beta.max(), beta.min())
            user_batch = test_users[start: end]
            user_batch = torch.tensor(user_batch).cuda()
            item_batch = torch.tensor(item_batch).cuda()

            rate_batch_tensor = self.model.backbone.predict_all(user_batch, item_batch)
            rate_batch = rate_batch_tensor.detach().cpu().numpy()
            rate_batch = np.array(rate_batch)
            # print(np.min(rate_batch), np.max(rate_batch), np.nanvar(rate_batch), np.mean(rate_batch))
            if opt.method == 'PD':
                rate_batch = self.elu(rate_batch)
            elif opt.method == 'PDA':
                rate_batch = self.elu(rate_batch) * \
                    (popularity_item_np ** opt.PDA_gamma)
            elif opt.is_debias:
                rate_batch = np.logaddexp(0, rate_batch)  # Activation function
                rate_batch = rate_batch * np.tanh(popularity_item_np)
            elif opt.method == 'p': # Use only popularity for recommendation
                popularity_item_np = np.load(opt.DICE_popularity_path)
                rate_batch = np.zeros(
                    rate_batch.shape) + np.tanh(popularity_item_np)
            elif opt.method == 'ratingBase':
                rate_batch = np.logaddexp(0, rate_batch)  # Activation function
                rate_batch = rate_batch * np.tanh(popularity_item_np)

            rate_batch = np.array(rate_batch)
            test_items = []  # Test set data
            test_items_h = []  # High-rated test set data
            user_batch = user_batch.cpu().numpy()
            item_batch = item_batch.cpu().numpy()

            for user in user_batch:
                test_items.append(self.test_set[user])
                if user in self.test_set_h.keys():
                    test_items_h.append(self.test_set_h[user])
                else:
                    test_items_h.append([])

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking
            # list.
            # user_indices = [idx for idx, user in enumerate(user_batch) if user in self.train_items.keys()]
            # train_items_off = [self.train_items[user] for user in user_batch if user in self.train_items.keys()]
            # for idx, items in zip(user_indices, train_items_off):
            #     rate_batch[idx, items] = -np.inf
            for idx, user in enumerate(user_batch):
                if user in self.train_items.keys():
                    train_items_off = self.train_items[user]
                    rate_batch[idx][train_items_off] = -np.inf
            batch_result = self.evaluate(
                rate_batch, test_items, test_items_h, max_top, top_at_k, user_batch)

            # batch_result = eval_score_matrix_foldout(rate_batch, test_items,
            # max_top)  # (B,k*metric_num), max_top= 20
            all_result.append(batch_result)

        # Concatenate the results of each batch and calculate the average for each user
        all_result = np.concatenate(all_result, axis=0)
        ratings = all_result[:, 4]

        final_result = np.nanmean(all_result, axis=0)  # mean

        # Update the results for each metric
        result['recall'] += final_result[0] if not np.isnan(final_result[0]) else np.nan
        result['map'] += final_result[1] if not np.isnan(final_result[1]) else np.nan
        result['ndcg'] += final_result[2] if not np.isnan(final_result[2]) else np.nan
        result['hit_rate'] += final_result[3] if not np.isnan(final_result[3]) else np.nan

        non_zero_count = np.sum(ratings != 0)
        if non_zero_count > 0:
            result['averageRating'] += np.sum(ratings) / non_zero_count
        else:
            result['averageRating'] += np.nan

        result['recall@k'] += final_result[5] if not np.isnan(final_result[5]) else np.nan
        result['map@k'] += final_result[6] if not np.isnan(final_result[6]) else np.nan
        result['ndcg@k'] += final_result[7] if not np.isnan(final_result[7]) else np.nan
        result['hit_rate@k'] += final_result[8] if not np.isnan(final_result[8]) else np.nan
        return result

    def run_test(self, epoch, no_learn=False):
        # Execute test() and process the output results
        users_to_test = list(self.test_set.keys())
        if opt.method == 'PDA':
            PDA_popularity = pd.read_csv(opt.PDA_popularity_path, sep='\t')
            popularity_item_np = PDA_popularity['8'] + \
                opt.PDA_alpha * (PDA_popularity['8'] - PDA_popularity['7'])
            popularity_item_np = np.maximum(popularity_item_np.values, 0)
        elif opt.is_debias:
            popularity_item_np = self.pbd.popularity(
                np.arange(
                    self.item_num), np.ones(
                    self.item_num,) * self.max_time)

            if opt.use_q:
                q = F.softplus(self.model.q).cpu().detach().numpy()
            if opt.use_a:
                a = F.softplus(self.model.a).cpu().detach().numpy()
            if opt.use_b:
                b = F.softplus(self.model.b).cpu().detach().numpy()

            if opt.useRatings:
                # ari = (torch.from_numpy(self.pbd.ariall()).cuda()).cpu().numpy()
                ari = self.pbd.ari(
                    np.arange(
                        self.item_num), np.ones(
                        self.item_num,) * self.max_time)

            # Calculate parameters to be used during inference
            if opt.is_debias:
                # Matching only
                if opt.method[-2:] == '-e':
                    popularity_item_np = np.ones(popularity_item_np.shape)
                # Standalone only
                elif opt.method in ['TIDE-noc', 'TIDE-C', 'TIDE-RinC-C', 'TIDE-ModeC-C', 'TIDE-counts-C'] or opt.method[-3:] == '-SC':
                    popularity_item_np = q
                elif opt.method in ['SbC-C', 'SbC-counts-C']:
                    popularity_item_np = ari
                elif opt.method in ['aSbC-noc', 'aSbC-C', 'aSbC-fixa-C', 'aSbC-counts-C']:
                    popularity_item_np = a * ari
                elif opt.method in ['TIDE-noq', 'aSbC-S']:
                    popularity_item_np = b * popularity_item_np
                # Combine two
                elif opt.method in ['qaSbC-C', 'qaSbC-counts-C', 'qaSbC-noc']:
                    popularity_item_np = q + a * ari
                elif opt.method in ['SbC', 'SbC-counts']:
                    popularity_item_np = ari + b * popularity_item_np
                elif opt.method in ['aSbC', 'aSbC-fixa', 'aSbC-counts']:
                    popularity_item_np = a * ari + b * popularity_item_np
                elif opt.method in ['TIDE', 'TIDE-full', 'TIDE-RinC', 'TIDE-ModeC', 'TIDE-counts', 'qaSbC-S']:
                    popularity_item_np = q + b * popularity_item_np
                # Combine three
                elif opt.method in ['qaSbC', 'qaSbC-counts']:
                    popularity_item_np = q + a * ari + b * popularity_item_np
                else:
                    raise ValueError('Invalid method name')

            if opt.is_debias and opt.show_performance:
                if opt.use_q:
                    str_tide_para_q = 'min(q) = %f, mean(q) = %f, max(q) = %f, var(q) = %f' % (
                        np.min(q), np.mean(q), np.max(q), np.nanvar(q))
                    util.print_str(opt.log_path, str_tide_para_q)
                if opt.use_a:
                    str_tide_para_a = 'min(a) = %f, mean(a) = %f, max(a) = %f, var(a) = %f' % (
                        np.min(a), np.mean(a), np.max(a), np.nanvar(a))
                    util.print_str(opt.log_path, str_tide_para_a)
                if opt.use_b:
                    str_tide_para_b = 'min(b) = %f, mean(b) = %f, max(b) = %f, var(b) = %f' % (
                        np.min(b), np.mean(b), np.max(b), np.nanvar(b))
                    util.print_str(opt.log_path, str_tide_para_b)

        elif opt.method == 'ratingBase':
            ari = self.pbd.ari(
                np.arange(
                    self.item_num), np.ones(
                    self.item_num,) * self.max_time)
            popularity_item_np = ari
        else:
            popularity_item_np = 0

        ret = self.test(users_to_test, popularity_item_np)
        perf_str1 = 'recall=[%.5f], map=[%.5f], ndcg=[%.5f], hit_rate=[%.5f]' % \
            (ret['recall'][0], ret['map'][0], ret['ndcg'][0], ret['hit_rate'][0])

        perf_str2 = 'recall@k=[%.5f], map@k=[%.5f], ndcg@k=[%.5f], hit_rate@k=[%.5f]' % \
            (ret['recall@k'][0], ret['map@k'][0], ret['ndcg@k'][0], ret['hit_rate@k'][0])

        if opt.show_performance:
            util.print_str(opt.log_path, perf_str1 + perf_str2)
            # util.print_str(opt.log_path, perf_str2)

        if no_learn:
            return ret

        save_model_flag = 0
        if opt.debug:
            if opt.high_rating_train:
                print("besrepock?", ret['recall@k'][0] + ret['map@k'][0] + ret['ndcg@k'][0] + ret['hit_rate@k'][0],
                        self.best_perf['recall@k'] + self.best_perf['map@k'] + self.best_perf['ndcg@k'] + self.best_perf['hit_rate@k'])
            else:
                print("besrepock?", ret['recall'][0] + ret['map'][0] + ret['ndcg'][0] + ret['hit_rate'][0],
                    self.best_perf['recall'] + self.best_perf['map'] + self.best_perf['ndcg'] + self.best_perf['hit_rate'])
            print("epoch", epoch)

        if opt.high_rating_train:
            if ret['recall@k'] + ret['map@k'] + ret['ndcg@k'] + ret['hit_rate@k'] > self.best_perf['recall@k'] + \
                    self.best_perf['map@k'] + self.best_perf['ndcg@k'] + self.best_perf['hit_rate@k']:
                self.record_best_performance(ret, epoch)
                save_model_flag = 1
        else:
            if ret['recall'] + ret['map'] + ret['ndcg'] + ret['hit_rate'] > self.best_perf['recall'] + \
                    self.best_perf['map'] + self.best_perf['ndcg'] + self.best_perf['hit_rate']:
                self.record_best_performance(ret, epoch)
                save_model_flag = 1

        return save_model_flag

    def record_best_performance(self, ret, epoch):
        """
        Records the best performance metrics achieved during evaluation.

        Parameters:
        ret (dict): A dictionary containing the performance metrics.
        epoch (int): The epoch number at which the best performance was achieved.
        """
        self.best_perf['recall'] = ret['recall']
        self.best_perf['map'] = ret['map']
        self.best_perf['ndcg'] = ret['ndcg']
        self.best_perf['hit_rate'] = ret['hit_rate']
        self.best_perf['best_epoch'] = epoch
        self.best_perf['recall@k'] = ret['recall@k']
        self.best_perf['map@k'] = ret['map@k']
        self.best_perf['ndcg@k'] = ret['ndcg@k']
        self.best_perf['hit_rate@k'] = ret['hit_rate@k']
        self.best_perf['averageRating'] = ret['averageRating']
