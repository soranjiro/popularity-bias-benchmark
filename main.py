from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import torch
import torch.optim as optim
import sys
import cppimport
import util
import optuna


import models
from evaluate import Evaluator
from config import opt
from data.dataset import Data

# import os
sys.path.append('cppcode')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    print("main() loading...", flush=True)
    pbd = cppimport.imp(opt.pbd_module)
    print("main() loaded")

    setup_seed(opt.seed)
    opt.get_input_path()
    train_data = pd.read_csv(opt.train_data, sep='\t')
    val_data = pd.read_csv(opt.val_data, sep='\t')
    test_data = pd.read_csv(opt.test_data, sep='\t')
    user_num = max(
        train_data['user'].max(),
        test_data['user'].max(),
        val_data['user'].max()) + 1
    item_num = max(
        train_data['item'].max(),
        test_data['item'].max(),
        val_data['item'].max()) + 1
    print(f"Number of users: {user_num}, Number of items: {item_num}, Total interactions: {len(train_data) + len(val_data) + len(test_data)}")
    min_time = train_data['timestamp'].min()
    max_time = train_data['timestamp'].max()
    num_train_interactions = len(train_data)

    # model = getattr(models, (opt.backbone + '_Model'))
    model = getattr(models, 'Models')
    train_dataset = Data(train_data, item_num, 4, True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0)

    model = model(
        user_num,
        item_num,
        64,
        min_time,
        max_time,
        num_train_interactions
    )
    model.cuda()
    optimizer = model.get_initializer()
    evaluator = Evaluator(model, user_num, item_num, max_time)

    if opt.val:
        pbd.load_user_interation_val()
    else:
        pbd.load_user_interation_test()

    # loss_str_flag = 0

    if opt.test_only:
        model.load_state_dict(torch.load(opt.model_path_to_load))
        if opt.save_params:
            if opt.use_q:
                torch.save(model.q, opt.q_path)
            if opt.use_a:
                torch.save(model.a, opt.a_path)
            if opt.use_b:
                torch.save(model.b, opt.b_path)
            # torch.save(model.state_dict(), opt.params_path)
        evaluator.run_test(0)
        return evaluator.best_perf

    # evaluator.run_test(0, popularity_item_PDA)
    DICE_alpha = 0.1 / 0.9
    for epoch in range(opt.epochs):
        if opt.debug:
            print("epoch:", epoch, flush=True)
        # os.mknod(opt.model_path + '/%s.pth' % epoch)
        DICE_alpha *= 0.9
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()  # During the training phase, this step generates actual training samples.
        loss_sum = torch.tensor([0]).cuda()
        batch_id = 0
        for user, item_i, item_j, timestamp, split_idx in train_loader:
            batch_id += 1
            if opt.debug:
                print(batch_id, flush=True, end=' ')
            # batch_t0 = time.time()
            user = user.cuda().long()
            item_i = item_i.cuda().long()
            item_j = item_j.cuda().long()
            timestamp = timestamp.cpu().numpy()
            split_idx = split_idx.cpu().numpy()
            # model.zero_grad()
            optimizer.zero_grad()
            if opt.method == 'IPS':
                loss = model(user, item_i, item_j, timestamp, split_idx)
            elif opt.method == 'DICE':
                loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss = \
                    model(user, item_i, item_j, timestamp, split_idx)
                loss = loss_click + DICE_alpha * \
                    (loss_interest + loss_popularity_1 + loss_popularity_2) + 0.01 * loss_discrepancy
                loss += opt.lamb * reg_loss
            else:
                prediction_i, prediction_j, reg_loss = model(
                    user, item_i, item_j, timestamp, split_idx)  # Call the forward() method
                # BPRloss
                # loss = -(prediction_i - prediction_j).sigmoid().log().sum()

                loss = torch.mean(torch.nn.functional.softplus(prediction_j - prediction_i))
                # print(loss, reg_loss)
                loss += opt.lamb * reg_loss

            # Perform backpropagation to compute the gradients of the loss with respect to model parameters
            loss.backward()  # Obtain gradients here
            # Update the model parameters using the computed gradients
            optimizer.step()  # Update parameters using gradients
            loss_sum = loss_sum + loss
            #
        if opt.debug:
            print('')
        # Training for one epoch is complete, start testing

        if epoch % 1 == 0:
            model.eval()  # During testing, use batch normalization values and disable dropout
            save_model_flag = evaluator.run_test(epoch)
            if save_model_flag:
                open(opt.model_path, 'w').close()
                torch.save(model.state_dict(), opt.model_path)
                # torch.save(model.state_dict(), 'model/' + opt.method + '/model.pth')
                if opt.show_performance:
                    print("saved model")
        elapsed_time = time.time() - start_time
        if opt.show_performance:
            with open(opt.log_path, 'a+') as f_log:
                if opt.method == 'DICE':
                    f_log.write('Loss:%.1f, %.1f, %.1f, %.1f, %.1f, %.1f'
                                % (loss.detach().cpu().numpy(),
                                   loss_click.detach().cpu().numpy(),
                                   loss_interest.detach().cpu().numpy(),
                                   loss_popularity_1.detach().cpu().numpy(),
                                   loss_popularity_2.detach().cpu().numpy(),
                                   loss_discrepancy.detach().cpu().numpy()))
                else:
                    f_log.write(
                        'Loss:%f\n' %
                        loss_sum.detach().cpu().numpy()[0])
                f_log.write(
                    "The time elapse of epoch {:03d}".format(epoch) +
                    " is: " +
                    time.strftime(
                        "%H: %M: %S",
                        time.gmtime(elapsed_time)) +
                    '\n')
            if opt.method == 'DICE':
                print('Loss:', loss.detach().cpu().numpy(),
                      loss_click.detach().cpu().numpy(),
                      loss_interest.detach().cpu().numpy(),
                      loss_popularity_1.detach().cpu().numpy(),
                      loss_popularity_2.detach().cpu().numpy(),
                      loss_discrepancy.detach().cpu().numpy())
            else:
                print('Loss:', loss_sum.detach().cpu().numpy()[0])
            print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
                  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        if epoch - evaluator.best_perf['best_epoch'] > 10: # and epoch > 50:
            with open(opt.log_path, 'a+') as f_log:
                f_log.write("early stop at %d epoch" % epoch)
            print("early stop at %d epoch" % epoch)
            break

    if opt.debug:
        print("Best Epoch:", evaluator.best_perf['best_epoch'], "type:", type(evaluator.best_perf['best_epoch']))
        print("Recall:", evaluator.best_perf['recall'][0])
        print('map:', evaluator.best_perf['map'][0])
        print("NDCG:", evaluator.best_perf['ndcg'][0])
        print('hit_rate:', evaluator.best_perf['hit_rate'][0])
        print("Average Rating:", evaluator.best_perf['averageRating'][0])

    with open(opt.log_path, 'a+') as f_log:
        f_log.write(
            "\nEnd. Best epoch {:03d}: recall = {:.5f}, map = {:.5f}, NDCG = {:.5f}, hit_rate = {:.5f}, "
            "sum = {:.5f}, averageRating = {:.5f}".format(
                int(evaluator.best_perf['best_epoch']),
                float(evaluator.best_perf['recall'][0]),
                float(evaluator.best_perf['map'][0]),
                float(evaluator.best_perf['ndcg'][0]),
                float(evaluator.best_perf['hit_rate'][0]),
                float(evaluator.best_perf['recall'][0] +
                evaluator.best_perf['map'][0] +
                evaluator.best_perf['ndcg'][0] +
                evaluator.best_perf['hit_rate'][0]),
                float(evaluator.best_perf['averageRating'][0])))
        '''f_log.write(" recall = {:.5f}, precision = {:.5f}, "
                    "NDCG = {:.5f}\n".format(evaluator.best_perf_h['recall'][0],
                                             evaluator.best_perf_h['precision'][0],
                                             evaluator.best_perf_h['ndcg'][0]))'''
        f_log.write(
            "recall@k = {:.5f}, map@k = {:.5f}, NDCG@k = {:.5f}, hit_rate@k = {:.5f}\n".format(
                evaluator.best_perf['recall@k'][0],
                evaluator.best_perf['map@k'][0],
                evaluator.best_perf['ndcg@k'][0],
                evaluator.best_perf['hit_rate@k'][0]))
    print(
        "End. Best epoch {:03d}: recall = {:.5f}, map = {:.5f}, NDCG = {:.5f}, hit_rate = {:.5f}, "
        "averageRating = {:.5f}".format(
            int(evaluator.best_perf['best_epoch']),
            float(evaluator.best_perf['recall'][0]),
            float(evaluator.best_perf['map'][0]),
            float(evaluator.best_perf['ndcg'][0]),
            float(evaluator.best_perf['hit_rate'][0]),
            float(evaluator.best_perf['averageRating'][0])))
    '''print(" recall = {:.5f}, precision = {:.5f}, "
          "NDCG = {:.5f}".format(evaluator.best_perf_h['recall'][0],
                                   evaluator.best_perf_h['precision'][0],
                                   evaluator.best_perf_h['ndcg'][0]))'''
    print(
        " recall@k = {:.5f}, map@k = {:.5f}, NDCG@k = {:.5f}, hit_rate@k = {:.5f}".format(
            evaluator.best_perf['recall@k'][0],
            evaluator.best_perf['map@k'][0],
            evaluator.best_perf['ndcg@k'][0],
            evaluator.best_perf['hit_rate@k'][0])
        )

    if opt.is_debias:
        def output_val_eval(method):
            opt.method = method
            print('opt.method:', opt.method)
            _ = evaluator.run_test(0, no_learn=True)

        if opt.method == 'TIDE':
            output_val_eval('TIDE-e')
            output_val_eval('TIDE-C')
            opt.method = 'TIDE'
        elif opt.method == 'SbC':
            output_val_eval('SbC-e')
            output_val_eval('SbC-C')
            opt.method = 'SbC'
        elif opt.method == 'aSbc':
            output_val_eval('aSbc-e')
            output_val_eval('aSbc-C')
            output_val_eval('aSbc-S')
            opt.method = 'aSbc'
        elif opt.method == 'qaSbC':
            output_val_eval('qaSbC-e')
            output_val_eval('qaSbC-C')
            output_val_eval('qaSbC-S')
            output_val_eval('qaSbC-SC')
            opt.method = 'qaSbC'
    return evaluator.best_perf


def test_after_training():
    opt.model_path_to_load = opt.model_path
    opt.test_only = True
    opt.val = False
    opt.get_input_path()

    def test():
        print(f'**********topk = {opt.topk}, top_at_k = {opt.top_at_k}**********')
        opt.method = opt.initialMethod
        print('---------testing----------')
        main()
        print(opt.method)

        if opt.is_debias and not opt.method.endswith('-C'):
            def test_methods(base_method, suffixes):
                for suffix in suffixes:
                    opt.method = f'{base_method}-{suffix}'
                    print(f'---------testing {opt.method}----------')
                    main()

            method_suffixes = {
                'TIDE': ['e', 'C'],
                'TIDE-QisARi': ['e', 'C'],
                'TIDE-QisARi-C': [],
                'TIDE-QisARiQi': ['e', 'S', 'C'],
                'TIDE-QisARiQi-C': [],
                'TIDE-QisARiQi-fixa': ['e', 'C'],
                'TIDE-QAddAlphaARi': ['e', 'S', 'C', 'SC'],
                'TIDE-QAddAlphaARi-C': [],
                'TIDE-counts': ['e', 'C'],
                'TIDE-QisARi-counts': ['e', 'C'],
                'TIDE-QisARiQi-counts': ['e', 'C'],
                'TIDE-QAddAlphaARi-counts': ['e', 'C', 'SC'],
                'TIDE-QAddAlphaARi-noc': ['e', 'SC'],
            }

            if opt.method in method_suffixes:
                test_methods(opt.method, method_suffixes[opt.method])
            else:
                raise ValueError(f'Unknown method: {opt.method}')


    topk_list = [1,3,5,8,10,15,20,25]
    opt.top_at_k = 3
    for topk in topk_list:
        opt.topk = topk
        test()

    top_at_k_list = [1,3,5,8,10,15,20,25]
    opt.topk = 20
    for top_at_k in top_at_k_list:
        opt.top_at_k = top_at_k
        test()

def main_base():
    lr_list = [opt.lr]
    lamb_list = [opt.lamb]
    lr_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    lamb_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

    # lr_list = [0.1]
    # lamb_list = [0.0001]

    # lr_list = [0.1]
    # lamb_list = [0.00001, 0.0001, 0.001]
    # lamb_list = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]

    # lr_list = [0.1, 1, 10]
    # lamb_list = [0.00001, 0.0001, 0.001]

    # lamb_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

    '''lr_list = [0.00003, 0.0001, 0.0003]
    lamb_list = [0.003, 0.01]'''

    best_para = {'lr': 0, 'lamb': 0}
    if opt.high_rating_train:
        best_perf = {'recall@k': np.zeros((1,)),
                     'map@k': np.zeros((1,)),
                     'ndcg@k': np.zeros((1,)),
                     'hit_rate@k': np.zeros((1,))}
    else:
        best_perf = {'recall': np.zeros((1,)),
                    'map': np.zeros((1,)),
                    'ndcg': np.zeros((1,)),
                    'hit_rate': np.zeros((1,))
                    }

    perf_df = pd.DataFrame(np.zeros((len(lamb_list), len(lr_list))),
                           columns=[str(j) for j in lr_list],
                           index=[str(j) for j in lamb_list])

    '''lr_list = [0.000001]
    lamb_list = [0.001]'''

    if (len(lr_list) == 1) and (len(lamb_list) == 1) or opt.no_search_hyper:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = True

        for opt.lamb in lamb_list:
            for opt.lr in lr_list:
                print('\n\n')
                str_para = 'lr = %f, lamb = %f' % (opt.lr, opt.lamb)
                util.print_str(opt.log_path, str_para)
                perf = main()

                if opt.high_rating_train:
                    perf_df[str(opt.lr)][str(opt.lamb)] = perf['recall@k'] + perf['map@k'] + perf['ndcg@k'] + perf['hit_rate@k']
                    if perf['recall@k'] + perf['map@k'] + perf['ndcg@k'] + perf['hit_rate@k'] > \
                            best_perf['recall@k'] + best_perf['map@k'] + best_perf['ndcg@k'] + best_perf['hit_rate@k']:
                        best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'] = \
                            perf['recall@k'], perf['map@k'], perf['ndcg@k'], perf['hit_rate@k']
                        best_para['lr'], best_para['lamb'] = opt.lr, opt.lamb
                else:
                    perf_df[str(opt.lr)][str(opt.lamb)] = perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate']
                    if perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate'] > \
                            best_perf['recall'] + best_perf['map'] + best_perf['ndcg'] + best_perf['hit_rate']:
                        best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'] = \
                            perf['recall'], perf['map'], perf['ndcg'], perf['hit_rate']
                        best_para['lr'], best_para['lamb'] = opt.lr, opt.lamb

        str_best_para = 'Best parameter: lr = %f, lamb = %f' % (best_para['lr'], best_para['lamb'])
        if opt.high_rating_train:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'])
        else:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'])
        util.print_str(opt.log_path, str_best_para)
        util.print_str(opt.log_path, str_best_perf)
        print(perf_df)

        opt.lr, opt.lamb = best_para['lr'], best_para['lamb']
        main()
        test_after_training()


def main_IPS():
    IPS_lambda_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    # IPS_lambda_list = [opt.IPS_lambda]

    best_para = {'IPS_lambda': 0}
    if opt.high_rating_train:
        best_perf = {
            'recall@k': np.zeros((1,)),
            'map@k': np.zeros((1,)),
            'ndcg@k': np.zeros((1,)),
            'hit_rate@k': np.zeros((1,))
        }
    else:
        best_perf = {
            'recall': np.zeros((1,)),
            'map': np.zeros((1,)),
            'ndcg': np.zeros((1,)),
            'hit_rate': np.zeros((1,))
        }

    if len(IPS_lambda_list) == 1 or opt.no_search_hyper:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = True
        outer_tide_start_time = time.time()
        for opt.IPS_lambda in IPS_lambda_list:
            print('\n\n')
            print('\nIPS_lambda = %d' % opt.IPS_lambda)
            with open(opt.log_path, 'a+') as f_log:
                f_log.write('\nIPS_lambda = %d\t' % opt.IPS_lambda)
            perf = main()

            if opt.high_rating_train:
                if perf['recall@k'] + perf['map@k'] + perf['ndcg@k'] + perf['hit_rate@k'] > \
                        best_perf['recall@k'] + best_perf['map@k'] + best_perf['ndcg@k'] + best_perf['hit_rate@k']:
                    best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'] = \
                        perf['recall@k'], perf['map@k'], perf['ndcg@k'], perf['hit_rate@k']
                    best_para['IPS_lambda'] = opt.IPS_lambda
            else:
                if perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate'] > \
                        best_perf['recall'] + best_perf['map'] + best_perf['ndcg'] + best_perf['hit_rate']:
                    best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'] = \
                        perf['recall'], perf['map'], perf['ndcg'], perf['hit_rate']
                    best_para['IPS_lambda'] = opt.IPS_lambda

        str_best_para = 'Best parameter: IPS_lambda = %d' % (best_para['IPS_lambda'])
        if opt.high_rating_train:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'])
        else:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'])
        util.print_str(opt.log_path, str_best_para)
        util.print_str(opt.log_path, str_best_perf)

        opt.IPS_lambda = best_para['IPS_lambda']
        main()
        test_after_training()
        outer_tide_elapsed_time = time.time() - outer_tide_start_time
        print("The time elapse is: " + time.strftime("%H: %M: %S", time.gmtime(outer_tide_elapsed_time)))


def main_PD():
    '''lr_list = [opt.lr]
    lamb_list = [opt.lamb]'''
    '''lr_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    lamb_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]'''

    PDA_gamma_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    # PDA_gamma_list = [opt.PDA_gamma]
    # PDA_gamma_list = [0.1]

    best_para = {'PDA_gamma': 0}
    if opt.high_rating_train:
        best_perf = {
            'recall@k': np.zeros((1,)),
            'map@k': np.zeros((1,)),
            'ndcg@k': np.zeros((1,)),
            'hit_rate@k': np.zeros((1,))
        }
    else:
        best_perf = {'recall': np.zeros((1,)),
                    'precision': np.zeros((1,)),
                    'ndcg': np.zeros((1,)),
                    'hit_rate': np.zeros((1,))
                    }

    perf_pd = pd.DataFrame(np.zeros((1, len(PDA_gamma_list))), columns=[str(j) for j in PDA_gamma_list])

    if len(PDA_gamma_list) == 1 or opt.no_search_hyper:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = True
        for opt.PDA_gamma in PDA_gamma_list:
            print('\n\n')
            str_para = 'lr = %f, lamb = %f, PDA_gamma = %.2f' % (
                  opt.lr, opt.lamb, opt.PDA_gamma)
            util.print_str(opt.log_path, str_para)
            perf = main()
            perf_pd[str(opt.PDA_gamma)] = perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate']
            if opt.high_rating_train:
                if perf['recall@k'] + perf['map@k'] + perf['ndcg@k'] + perf['hit_rate@k'] > \
                        best_perf['recall@k'] + best_perf['map@k'] + best_perf['ndcg@k'] + best_perf['hit_rate@k']:
                    best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'] = \
                        perf['recall@k'], perf['map@k'], perf['ndcg@k'], perf['hit_rate@k']
                    best_para['PDA_gamma'] = opt.PDA_gamma
            else:
                if perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate'] > \
                        best_perf['recall'] + best_perf['map'] + best_perf['ndcg'] + best_perf['hit_rate']:
                    best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'] = \
                        perf['recall'], perf['map'], perf['ndcg'], perf['hit_rate']
                    best_para['PDA_gamma'] = opt.PDA_gamma

        str_best_para = 'Best parameter: PDA_gamma = %f' % (best_para['PDA_gamma'])
        if opt.high_rating_train:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'])
        else:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'])
        util.print_str(opt.log_path, str_best_para)
        util.print_str(opt.log_path, str_best_perf)
        print(perf_pd)

        opt.PDA_gamma = best_para['PDA_gamma']
        main()
        test_after_training()


def main_PDA():
    # PDA_gamma_list = [opt.PDA_gamma]
    # PDA_alpha_list = [opt.PDA_alpha]
    PDA_gamma_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    # PDA_gamma_list.reverse()
    PDA_alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # PDA_alpha_list = [0.35, 0.4]

    best_para = {'PDA_gamma': 0, 'PDA_alpha': 0}
    if opt.high_rating_train:
        best_perf = {
            'recall@k': np.zeros((1,)),
            'map@k': np.zeros((1,)),
            'ndcg@k': np.zeros((1,)),
            'hit_rate@k': np.zeros((1,))
        }
    else:
        best_perf = {'recall': np.zeros((1,)),
                     'precision': np.zeros((1,)),
                     'ndcg': np.zeros((1,)),
                     'hit_rate': np.zeros((1,))
        }

    perf_pd = pd.DataFrame(np.zeros((len(PDA_alpha_list), len(PDA_gamma_list))),
                           columns=[str(j) for j in PDA_gamma_list],
                           index=[str(j) for j in PDA_alpha_list])

    if (len(PDA_gamma_list) == 1) and (len(PDA_alpha_list) == 1) or opt.no_search_hyper:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = True
        for opt.PDA_gamma in PDA_gamma_list:
            for opt.PDA_alpha in PDA_alpha_list:
                print('\n\n')
                str_para = 'lr = %f, lamb = %f, PDA_gamma = %.2f, PDA_alpha = %.2f' % (
                      opt.lr, opt.lamb, opt.PDA_gamma, opt.PDA_alpha)
                util.print_str(opt.log_path, str_para)
                perf = main()
                perf_pd[str(opt.PDA_gamma)][str(opt.PDA_alpha)] = perf['ndcg']
                if opt.high_rating_train:
                    if perf['recall@k'] + perf['map@k'] + perf['ndcg@k'] + perf['hit_rate@k'] > \
                            best_perf['recall@k'] + best_perf['map@k'] + best_perf['ndcg@k'] + best_perf['hit_rate@k']:
                        best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'] = \
                            perf['recall@k'], perf['map@k'], perf['ndcg@k'], perf['hit_rate@k']
                        best_para['PDA_gamma'], best_para['PDA_alpha'] = opt.PDA_gamma, opt.PDA_alpha
                else:
                    if perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate'] > \
                            best_perf['recall'] + best_perf['map'] + best_perf['ndcg'] + best_perf['hit_rate']:
                        best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'] = \
                            perf['recall'], perf['map'], perf['ndcg'], perf['hit_rate']
                        best_para['PDA_gamma'], best_para['PDA_alpha'] = opt.PDA_gamma, opt.PDA_alpha

        str_best_para = 'Best parameter: PDA_gamma = %f, PDA_alpha = %f' % (best_para['PDA_gamma'], best_para['PDA_alpha'])
        if opt.high_rating_train:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall@k'], best_perf['map@k'], best_perf['ndcg@k'], best_perf['hit_rate@k'])
        else:
            str_best_perf = 'Best performance: rec = %.5f, map = %.5f, ndcg = %.5f, hit_rate = %.5f' % (
                best_perf['recall'], best_perf['map'], best_perf['ndcg'], best_perf['hit_rate'])
        util.print_str(opt.log_path, str_best_para)
        util.print_str(opt.log_path, str_best_perf)
        print(perf_pd)

        opt.PDA_gamma = best_para['PDA_gamma']
        opt.PDA_alpha = best_para['PDA_alpha']
        main()
        test_after_training()


def main_debias():
    print("main_debias()", flush=True)

    if opt.no_search_hyper:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
        return

    # # filepath: /app/main.py
    def objective(trial):
        # 提案されたハイパーパラメータの範囲を定義
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lamb = trial.suggest_float("lamb", 1e-4, 1e-2, log=True)
        # lr = 0.01
        # lamb = 0.01
        if opt.use_q:
            q = trial.suggest_int("q", -5, -1)
            lr_q = trial.suggest_float("lr_q", 1e-4, 1e-1, log=True)
        if opt.use_a:
            a = trial.suggest_int("a", -5, -1)
            lr_a = trial.suggest_float("lr_a", 1e-4, 1e-1, log=True)
        if opt.use_b:
            b = trial.suggest_int("b", -5, -1)
            lr_b = trial.suggest_float("lr_b", 1e-4, 1e-1, log=True)

        if opt.backbone ==  'LightGCN':
            opt.n_layers = trial.suggest_int("n_layers", 2, 4)

        # Set values to the opt object
        opt.lr = lr
        opt.lamb = lamb
        if opt.use_q:
            opt.q = q
            opt.lr_q = lr_q
        if opt.use_a:
            opt.a = a
            opt.lr_a = lr_a
        if opt.use_b:
            opt.b = b
            opt.lr_b = lr_b

        # Train and evaluate the model
        perf = main()  # The main() function needs to be modified to return evaluation metrics

        # Return the evaluation metric to optimize
        if opt.high_rating_train:
            return perf['recall@k'] + perf['map@k'] + perf['ndcg@k'] + perf['hit_rate@k']
        else:
            return perf['recall'] + perf['map'] + perf['ndcg'] + perf['hit_rate']

    print("main_optuna()", flush=True)

    # # Create an Optuna study
    # study = optuna.create_study(direction="maximize")  # Aim for maximization
    # study.optimize(objective, n_trials=1000)  # Specify the number of trials

    # Create an Optuna study using SQLite
    storage_url = "sqlite:///optuna.db"  # Specify the SQLite database file
    train = "high_rate" if opt.high_rating_train else "click"
    study_name = f"study_{opt.dataset}_{opt.backbone}_{train}_{opt.method}_{opt.seed}"  # Dynamically generate to avoid conflicts
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,  # Reuse existing study if available
    )
    study.optimize(objective, n_trials=100, n_jobs=1)  # Enable parallel execution

    # Display the best parameters
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Retrain the model with the best parameters and test
    opt.lr = trial.params['lr']
    opt.lamb = trial.params['lamb']
    if opt.use_q:
        opt.q = trial.params['q']
        opt.lr_q = trial.params['lr_q']
    if opt.use_a:
        opt.a = trial.params['a']
        opt.lr_a = trial.params['lr_a']
    if opt.use_b:
        opt.b = trial.params['b']
        opt.lr_b = trial.params['lr_b']
    main()
    test_after_training()


# Execute a specific method and search for the best parameters (optional)
outer_all_start_time = time.time()
if opt.method == 'IPS':
    main_IPS()
elif opt.method == 'PD':
    main_PD()
elif opt.method == 'PDA':
    main_PDA()
elif opt.is_debias:
    main_debias()
else:
    print("opt.method = ", opt.method, flush=True)
    main_base()
outer__all_elapsed_time = time.time() - outer_all_start_time
print("The time elapse of all is: " + time.strftime("%H: %M: %S", time.gmtime(outer__all_elapsed_time)))


print("all done")
