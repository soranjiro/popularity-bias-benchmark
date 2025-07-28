import time
import os

import util

import numpy as np

class DefaultConfig(object):
    """
    Defines all parameters, including which dataset to use, which method to select,
    paths to required files, and parameters corresponding to each method.
    """
    def __init__(self):
        print("config__init__")

        # seed
        # 10, 20, 30, 40, 50
        self.seed = 20

        # backbone
        # 'LightGCN', 'MF', 'NCF'
        self.backbone = 'LightGCN'

        # dataset
        # 'Amazon-Baby_Products', 'Amazon-Music', 'Amazon-Software',
        # 'Amazon-Toys_and_Games', 'Amazon-Video_Games', 'Ciao',
        # 'Douban-book', 'Douban-music', 'MovieLens-1M',
        # 'Yelp_PA' is Yelp(Philadelphia)
        self.dataset = 'Amazon-Baby_Products'

        # methods
        # 'base', 'IPS', 'DICE', 'PD', 'PDA', 'TIDE', 'SbC', 'aSbC', 'qaSbC'
        self.initialMethod = 'TIDE'

        # directory to save parameters and recommend items
        self.dir = 'result'

        # whether to search hyperparameters
        self.no_search_hyper = True

        # training with high ratings
        self.high_rating_train = True

        # whether to save model and results
        self.is_save_model = False
        self.save_params = False
        self.save_recommend = False

        self.set_pbd_module()

        # whether to normalize ratings
        self.normalizedRatings = True

        self.method = self.initialMethod
        self.set_method_config()

        self.val = True
        self.test_only = False
        self.show_performance = True

        # Whether to enforce data preprocessing
        self.data_process = False

        self.debug = False

        self.top_at_k = 3

        self.model_path_to_load = 'model/' + self.dataset + '/' + \
            '2022-03-17 20.41.18_MF_TIDE' + '.pth'
        self.get_file_path()
        self.print_head_line()
        self.get_para()
        self.print_config_info()
        self.get_input_path()
        print("config__init__ end")

    def set_pbd_module(self):
        self.pbd_module = 'pybind_' + self.dataset

        if self.dataset == 'Amazon-Baby_Products':
            self.pbd_module = 'pybind_amazon_baby_products'
        elif self.dataset == 'Amazon-Music':
            self.pbd_module = 'pybind_amazon_music'
        elif self.dataset == 'Amazon-Software':
            self.pbd_module = 'pybind_amazon_software'
        elif self.dataset == 'Amazon-Toys_and_Games':
            self.pbd_module = 'pybind_amazon_toys_and_games'
        elif self.dataset == 'Amazon-Video_Games':
            self.pbd_module = 'pybind_amazon_video_games'

        elif self.dataset == 'Ciao':
            self.pbd_module = 'pybind_ciao'

        elif self.dataset == 'Douban-book':
            self.pbd_module = 'pybind_douban_book'
        elif self.dataset == 'Douban-music':
            self.pbd_module = 'pybind_douban_music'

        elif self.dataset == 'MovieLens-1M':
            self.pbd_module = 'pybind_movielens_1m'

        elif self.dataset == 'Yelp_PA':
            self.pbd_module = 'pybind_yelp_pa'
        else:
            raise ValueError('Invalid dataset')

    def set_method_config(self):
        self.use_q = False
        self.use_a = False
        self.use_b = False
        self.useRatings = False
        self.use_cti = True
        self.is_debias = False

        if self.method in ['TIDE', 'TIDE-noc', 'TIDE-noq', 'TIDE-e', 'TIDE-C']:
            self.use_q = True
            self.use_b = True
            self.is_debias = True
        elif self.method in ['TIDE-counts', 'TIDE-counts-e', 'TIDE-counts-C']:
            self.use_q = True
            self.use_b = True
            self.use_cti = False
            self.is_debias = True
        elif self.method in ['SbC', 'SbC-e', 'SbC-C']:
            self.use_b = True
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['SbC-counts', 'SbC-counts-e', 'SbC-counts-C']:
            self.use_b = True
            self.use_cti = False
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['aSbC', 'aSbC-e', 'aSbC-C', 'aSbC-S', 'aSbC-fixa', 'aSbC-fixa-e', 'aSbC-fixa-C', 'aSbC-fixa-S']:
            self.use_a = True
            self.use_b = True
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['aSbC-counts', 'aSbC-counts-e', 'aSbC-counts-C', 'aSbC-counts-S']:
            self.use_a = True
            self.use_b = True
            self.use_cti = False
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['qaSbC', 'qaSbC-e', 'qaSbC-C', 'qaSbC-S', 'qaSbC-SC']:
            self.use_q = True
            self.use_a = True
            self.use_b = True
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['qaSbC-counts', 'qaSbC-counts-e', 'qaSbC-counts-C', 'qaSbC-counts-S', 'qaSbC-counts-SC']:
            self.use_q = True
            self.use_a = True
            self.use_b = True
            self.use_cti = False
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['qaSbC-noc', 'qaSbC-noc-e', 'qaSbC-noc-C', 'qaSbC-noc-S', 'qaSbC-noc-SC']:
            self.use_q = True
            self.use_a = True
            self.useRatings = True
            self.is_debias = True
        elif self.method in ['base', 'IPS', 'DICE', 'PD', 'PDA']:
            self.use_q = False
            self.use_a = False
            self.use_b = False
            self.useRatings = False
        elif self.method in ['ratingBase']:
            self.use_q = False
            self.use_a = False
            self.use_b = False
            self.useRatings = True
        else:
            raise ValueError('Invalid method in config.py')

    def get_file_path(self):
        """
        To create file path for log and model saving
        """
        str_time = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())

        log_dir = 'log/' + self.dataset
        model_dir = 'model/' + self.dataset
        self.make_dir(log_dir)
        self.make_dir(model_dir)

        log_path = log_dir + '/' + str_time + '_' + self.backbone + "_" + self.method
        if not self.val:
            log_path += '_test'
        log_path += '.txt'

        if self.high_rating_train:
            train = 'high_rate'
        else:
            train = 'click'

        model_path = model_dir + '/' + str_time + '_' + \
            self.backbone + "_" + train + "_" + \
                self.method + '.pth'

        param_dir = "perf/%s/data/param" % self.dir
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        q_path = param_dir + "/q" + "-%s" % train +"-%s" % self.method + "-%s" % self.backbone + "-%s" % self.dataset
        a_path = param_dir + "/a" + "-%s" % train +"-%s" % self.method + "-%s" % self.backbone + "-%s" % self.dataset
        b_path = param_dir + "/b" + "-%s" % train +"-%s" % self.method + "-%s" % self.backbone + "-%s" % self.dataset
        params_path = param_dir + "/params" + "-%s" % train +"-%s" % self.method + "-%s" % self.backbone + "-%s" % self.dataset

        lgcn_graph_dir = './lgcn_graph/' + self.dataset
        self.make_dir(lgcn_graph_dir)
        graph_index_path = lgcn_graph_dir + '/lgcn_graph_index.npy'
        graph_data_path = lgcn_graph_dir + '/lgcn_graph_data.npy'

        self.log_path, self.model_path, self.q_path, self.a_path, self.b_path, self.params_path, \
            self.graph_index_path, self.graph_data_path = \
            log_path, model_path, q_path, a_path, b_path, params_path, graph_index_path, graph_data_path

        recommend_dir = 'perf/%s/data/recommend' % self.dir
        if not os.path.exists(recommend_dir):
            os.makedirs(recommend_dir)
        self.recommend_path = (
            recommend_dir
            + "/%s" % train
            + "_%s" % self.method
            + "_%s" % self.backbone
            + "_%s" % self.dataset
            + ".txt"
        )
        return log_path, model_path, q_path, graph_index_path, graph_data_path

    def make_dir(self, path):
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.makedirs(path)

    def print_head_line(self):
        if self.test_only:
            model_path_str = 'model_path = %s' % self.model_path_to_load
            self.print_str(model_path_str, window=False)

        method_str = 'backbone = %s, method = %s, dataset = %s, val = %s' % (
            self.backbone, self.method, self.dataset, self.val)
        self.print_str(method_str)

    def print_config_info(self):
        basic_info = 'lr = %.4f, lamb = %f, batch_size = %d, topk = %d, seed = %d' % (
            self.lr, self.lamb, self.batch_size, self.topk, self.seed)
        self.print_str(basic_info)

        if self.backbone == 'LightGCN':
            layer_info = 'n_layers = %d' % self.n_layers
            self.print_str(layer_info)
        elif self.backbone == 'NCF':
            layer_info = 'n_layers = %s' % self.n_layers
            self.print_str(layer_info)

        if self.method in ['base', 'ratingBase', 'DICE']:
            pass
        elif self.method == 'IPS':
            ips_info = 'IPS_lambda = %d' % self.IPS_lambda
            self.print_str(ips_info)

        elif self.method == 'PD':
            pd_info = 'PD_gamma = %.2f' % self.PDA_gamma
            self.print_str(pd_info)

        elif self.method == 'PDA':
            pda_info = 'PDA_gamma = %.2f, PDA_alpha = %.2f' % (
                self.PDA_gamma, self.PDA_alpha)
            self.print_str(pda_info)

        elif self.is_debias:
            if self.use_q and self.use_a and self.use_b:
                tide_info = 'tau = %d, q = %.2f, a = %.2f, b = %.2f\n' \
                            ' lr_q = %f, lr_a = %f, lr_b = %f' % (
                                self.tau, self.q, self.a, self.b, self.lr_q, self.lr_a, self.lr_b)
                self.print_str(tide_info)
            elif (self.use_q) and self.use_a and self.use_b:
                tide_info = 'tau = %d, a = %.2f, b = %.2f\n' \
                            ' lr_a = %f, lr_b = %f' % (
                                self.tau, self.a, self.b, self.lr_a, self.lr_b)
                self.print_str(tide_info)
            elif self.use_q and (not self.use_a) and self.use_b:
                tide_info = 'tau = %d, q = %.2f, b = %.2f\n' \
                            ' lr_q = %f, lr_b = %f' % (
                                self.tau, self.q, self.b, self.lr_q, self.lr_b)
                self.print_str(tide_info)
            elif self.use_q and self.use_a and (not self.use_b):
                tide_info = 'tau = %d, q = %.2f, a = %.2f\n' \
                            ' lr_q = %f, lr_a = %f' % (
                                self.tau, self.q, self.a, self.lr_q, self.lr_a)
                self.print_str(tide_info)
            elif self.use_b:
                tide_info = 'tau = %d, b = %.2f\n' \
                            ' lr_b = %f' % (
                                self.tau, self.b, self.lr_b)
                self.print_str(tide_info)
            else:
                raise ValueError('Invalid method in config.py')
        else:
            raise ValueError('Invalid method in config.py')

    def print_str(self, str_to_print, file=True, window=True):
        """
        To print string to log file and the window
        """
        util.print_str(self.log_path, str_to_print, file, window)

    def get_data_process_para(self):
        if self.dataset in [
            'Ciao',
            'Amazon-Music',
            'MovieLens-1M',
            ]:
            self.n_u_f, self.n_i_f = 5, 5
        elif self.dataset in [
            'Amazon-Baby_Products',
            'Amazon-Software',
            'Amazon-Toys_and_Games',
            'Amazon-Video_Games',
            'Yelp_PA',
            ]:
            self.n_u_f, self.n_i_f = 10, 10
        elif self.dataset in [
            'Douban-book',
            'Douban-music',
            ]:
            self.n_u_f, self.n_i_f = 20, 20
        else:
            self.n_u_f, self.n_i_f = 10, 10

    def get_base_para_lgcn(self):
        self.lr = 0.01
        self.lamb = 0.01
        self.topk = 20
        self.epochs = 200
        if self.dataset in ['Douban-book', 'Douban-music', 'MovieLens-1M', 'Yelp_PA']:
            self.batch_size = 8192
        else:
            self.batch_size = 2048
        self.n_layers = 3

    def get_base_para_mf(self):
        self.lr = 0.01
        self.lamb = 0.01
        self.topk = 20
        self.epochs = 200
        self.batch_size = 2048

    def get_base_para_ncf(self):
        self.lr = 0.01
        self.lamb = 0.01
        self.topk = 20
        self.epochs = 200
        self.batch_size = 1024
        self.n_layers = [64, 32, 16, 8]

    def get_basic_para(self):
        if self.backbone == 'LightGCN':
            self.get_base_para_lgcn()
        elif self.backbone == 'MF' or self.backbone == 'FM':
            self.get_base_para_mf()
        elif self.backbone == 'NCF':
            self.get_base_para_ncf()

    def get_ips_para_lgcn(self):
        # Set the lambda value based on the dataset
        if self.dataset in ["hoge_dataset"]:
            self.IPS_lambda = 30
        else:
            self.IPS_lambda = 30

    def get_ips_para_mf(self):
        # Set the lambda value based on the dataset
        if self.dataset in ["hoge_dataset"]:
            self.IPS_lambda = 30
        else:
            self.IPS_lambda = 30

    def get_ips_para_ncf(self):
        # Set the lambda value based on the dataset
        if self.dataset in ["hoge_dataset"]:
            self.IPS_lambda = 30
        else:
            self.IPS_lambda = 30

    def get_ips_para(self):
        if self.backbone == 'LightGCN':
            self.get_ips_para_lgcn()
        elif self.backbone == 'MF' or self.backbone == 'FM':
            self.get_ips_para_mf()
        else:
            self.get_ips_para_ncf()

    def get_tide_para_lgcn(self):
        # initial parameter if not using hyperparameter search
        self.lr = 0.01
        self.lamb = 0.01
        self.q = 0.0
        self.a = 0.0
        self.b = 0.0
        self.lr_q = 0.01
        self.lr_a = 0.01
        self.lr_b = 0.01
        # hyperparameter tau
        self.tau = 1 * pow(10, 7)
        self.lamb_q = 0
        self.lamb_a = 0
        self.lamb_b = 0

    def get_tide_para_mf(self):
        # initial parameter if not using hyperparameter search
        self.lr = 0.01
        self.lamb = 0.01
        self.q = 0.0
        self.a = 0.0
        self.b = 0.0
        self.lr_q = 0.01
        self.lr_a = 0.01
        self.lr_b = 0.01
        # hyperparameter tau
        self.tau = 1 * pow(10, 7)
        self.lamb_q = 0
        self.lamb_a = 0
        self.lamb_b = 0

    def get_tide_para_ncf(self):
        # initial parameter if not using hyperparameter search
        self.lr = 0.01
        self.lamb = 0.01
        self.q = 0.0
        self.a = 0.0
        self.b = 0.0
        self.lr_q = 0.01
        self.lr_a = 0.01
        self.lr_b = 0.01
        # hyperparameter tau
        self.tau = 1 * pow(10, 7)
        self.lamb_q = 0
        self.lamb_a = 0
        self.lamb_b = 0

    def get_tide_para(self):
        if self.backbone == 'LightGCN':
            self.get_tide_para_lgcn()
        elif self.backbone == 'MF' or self.backbone == 'FM':
            self.get_tide_para_mf()
        elif self.backbone == 'NCF':
            self.get_tide_para_ncf()

        if not self.no_search_hyper and self.method[:4] == 'TIDE':
            self.lr = 0.01
            self.lamb = 0.01

        if self.method == 'TIDE-fixq':
            self.lr_q = 0
        elif self.method in ['TIDE-QisARiQi-fixa', 'TIDE-QisARiQi-fixa-e', 'TIDE-QisARiQi-fixa-int']:
            self.lr_a = 0

    def get_pda_para_lgcn(self):
        if self.method == "PD":
            self.PDA_gamma = 0.02

        elif self.method == "PDA":
            self.PDA_gamma = 0.02
            self.PDA_alpha = 0.1

    def get_pda_para_mf(self):
        if self.method == "PD":
            self.PDA_gamma = 0.02

        elif self.method == "PDA":
            self.PDA_gamma = 0.02
            self.PDA_alpha = 0.1

    def get_pda_para_ncf(self):
        if self.method == "PD":
            self.PDA_gamma = 0.02

        elif self.method == "PDA":
            self.PDA_gamma = 0.02
            self.PDA_alpha = 0.1

    def get_pda_para(self):
        if self.backbone == 'LightGCN':
            self.get_pda_para_lgcn()
        elif self.backbone == 'MF' or self.backbone == 'FM':
            self.get_pda_para_mf()
        else:
            self.get_pda_para_ncf()

    def get_para(self):
        if self.use_q:
            self.q = 0
            self.lr_q = 0.001
        if self.use_a:
            self.a = 0
            self.lr_a = 0.001
        if self.use_b:
            self.b = 0
            self.lr_b = 0.001
        self.get_data_process_para()
        self.get_basic_para()
        self.get_ips_para()
        self.get_pda_para()
        self.get_tide_para()

    def get_input_path(self):
        main_path = 'data/'
        self.train_data = main_path + '{}/train_data.csv'.format(self.dataset)
        self.train_list = main_path + '{}/train_list.txt'.format(self.dataset)
        self.val_data = main_path + '{}/val_data.csv'.format(self.dataset)
        self.test_data = main_path + '{}/test_data.csv'.format(self.dataset)
        self.PDA_popularity_path = main_path + \
            '{}/PDA_popularity.csv'.format(self.dataset)
        self.DICE_popularity_path = main_path + \
            '{}/DICE_popularity.npy'.format(self.dataset)
        if self.val:
            self.test_list = main_path + "{}/val_list.txt".format(self.dataset)
            self.test_list_h = main_path + \
                '{}/val_high_rating_list.txt'.format(self.dataset)
            self.test_list_rating = main_path + "{}/val_list_rating.txt".format(
                self.dataset
            )
        else:  # test
            self.test_list = main_path + "{}/test_list.txt".format(self.dataset)
            self.test_list_h = (
                main_path + "{}/test_high_rating_list.txt".format(self.dataset)
            )
            self.test_list_rating = (
                main_path + "{}/test_list_rating.txt".format(self.dataset)
            )


opt = DefaultConfig()
