# coding=utf-8
# データを読み込み、有効なユーザーとアイテムのリストをフィルタリング
import numpy as np
import pandas as pd
import math
import sys
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

sys.path.append('../')
from config import opt


def process(dataset):
    print("1  filtering users and items")
    data_path = 'data/' + dataset + '/ratings.csv'

    f = open(data_path, 'r')
    lines = f.readlines()
    user_dict, item_dict = {}, {}
    dt2010 = datetime(2010, 1, 1)
    timestamp2010 = dt2010.timestamp()
    dt2014 = datetime(2014, 1, 1)
    timestamp2014 = dt2014.timestamp()
    dt2023 = datetime(2023, 1, 1)
    timestamp2023 = dt2023.timestamp()
    for line in lines:
        temp = line.strip().split(',')
        u_id = temp[0]
        i_id = temp[1]
        timestamp = float(temp[-1])
        if ('Amazon' in opt.dataset) and (opt.dataset not in ['Amazon-Music']):
            if timestamp > timestamp2014 and timestamp < timestamp2023:
                if u_id in user_dict.keys():
                    user_dict[u_id].append(i_id)
                else:
                    user_dict[u_id] = [i_id]

                if i_id in item_dict.keys():
                    item_dict[i_id].append(u_id)
                else:
                    item_dict[i_id] = [u_id]
        else:
            if u_id in user_dict.keys():
                user_dict[u_id].append(i_id)
            else:
                user_dict[u_id] = [i_id]

            if i_id in item_dict.keys():
                item_dict[i_id].append(u_id)
            else:
                item_dict[i_id] = [u_id]

    f_user_dict, f_item_dict = {}, {}

    n_u_f, n_i_f = opt.n_u_f, opt.n_i_f
    print(f'filtering users and items with n_u_f={n_u_f}, n_i_f={n_i_f}')
    print('n_users\tn_items')
    while True:
        print(len(user_dict.keys()), len(item_dict.keys()))
        flag1, flag2 = True, True

        for u_id in user_dict.keys():
            pos_items = user_dict[u_id]
            val_items = [idx for idx in pos_items if idx in item_dict.keys()]

            if len(val_items) >= n_u_f:
                f_user_dict[u_id] = val_items
            else:
                flag1 = False

        user_dict = f_user_dict.copy()

        for i_id in item_dict.keys():
            pos_users = item_dict[i_id]
            val_users = [udx for udx in pos_users if udx in user_dict.keys()]

            if len(pos_users) >= n_i_f:
                f_item_dict[i_id] = val_users
            else:
                flag2 = False

        item_dict = f_item_dict.copy()
        f_user_dict, f_item_dict = {}, {}

        if flag1 and flag2:
            print('filter done.')
            break

    # filtering
    all_data = pd.read_csv(data_path, sep=',', header=None, names=['user', 'item', 'rating', 'timestamp'], dtype={'user':object, 'item':object})
    interactions_num_1 = len(all_data)
    all_data = all_data[all_data['user'].isin(user_dict.keys())].reset_index(drop=True)
    all_data = all_data[all_data['item'].isin(item_dict.keys())].reset_index(drop=True)
    if opt.dataset in ['Douban-movie', 'Douban-movie_no_dup']:
        all_data = all_data[all_data['timestamp'] > timestamp2010].reset_index(drop=True)
    if ('Amazon' in opt.dataset) and (opt.dataset not in ['Amazon-CDs_and_Vinyl', 'Amazon-Music', 'Amazon-Movies_and_TV', 'Amazon-CDs_and_Vinyl_no_dup', 'Amazon-Music_no_dup', 'Amazon-Movies_and_TV_no_dup']):
        all_data = all_data[(all_data['timestamp'] > timestamp2014) & (all_data['timestamp'] < timestamp2023)].reset_index(drop=True)
    interactions_num_2 = len(all_data)
    print('interactions_num:\n', interactions_num_1, interactions_num_2)

    # Remapping IDs
    le = LabelEncoder()
    le.fit(all_data['user'].values)
    all_data['user'] = le.transform(all_data['user'].values)
    le.fit(all_data['item'].values)
    all_data['item'] = le.transform(all_data['item'].values)
    all_data = all_data.sort_values(by=['user', 'item']).reset_index(drop=True)
    # print('after ID remap:\n',all_data)

    user_num = all_data['user'].max() + 1
    item_num = all_data['item'].max() + 1

    min_time = all_data['timestamp'].min()
    max_time = all_data['timestamp'].max()

    # calculate PDA popularity
    K = 10
    time_interval = math.ceil((max_time - min_time) / K)
    PDA_popularity = pd.DataFrame(index=np.arange(item_num))
    data_concat = pd.DataFrame()
    for i in range(K):
        time_split_min = min_time + time_interval * i
        time_split_max = time_split_min + time_interval
        if i == K - 1:
            time_split_max = all_data['timestamp'].max() + 1
        data_split = all_data[
            (all_data['timestamp'] >= time_split_min) & (all_data['timestamp'] < time_split_max)].reset_index(drop=True)
        # print(time_split_min, time_split_max, len(data_split))
        count = data_split.item.value_counts()
        pop = pd.DataFrame(count)
        pop.columns = [str(i)]
        # (pop, pop.values.max())
        pop = pop / pop.values.max()
        PDA_popularity = pd.merge(PDA_popularity, pop, left_index=True, right_index=True, how='left').fillna(0)
        data_split['split_idx'] = i
        data_concat = pd.concat([data_concat, data_split]).reset_index(drop=True)
    all_data = data_concat
    PDA_popularity.to_csv('data/' + dataset + '/PDA_popularity.csv', index=0, sep='\t')

    # Calculate DICE and IPS popularity
    count = all_data.item.value_counts()
    pop = count / count.max()
    np.save(opt.DICE_popularity_path, pop)

    # Split training set, validation set, and test set
    print("4  split training set and test set")
    all_data = all_data.sort_values(by=['timestamp']).reset_index(drop=True)
    time_interval = math.ceil((max_time - min_time) / K)
    train_time_max = min_time + time_interval * (K - 1)
    train_data = all_data[all_data['timestamp'] < train_time_max]
    rest_data = all_data[all_data['timestamp'] >= train_time_max].reset_index(drop=True)
    val_user_set = np.random.choice(np.arange(user_num), int(user_num / 2), replace=False)
    test_user_set = np.setdiff1d(np.arange(user_num), val_user_set)

    def clean_user(data, user_set):
        """
        - Remove users whose ratings are all 5
        - Remove users who do not have any ratings of 5
        """
        user_ratings = data.groupby('user')['rating'].apply(list).reset_index()
        users_to_remove = []
        for _, row in user_ratings.iterrows():
            user_id = row['user']
            ratings = row['rating']
            if all(rating == 5 for rating in ratings) or not any(rating == 5 for rating in ratings):
                users_to_remove.append(user_id)
        return data[~data['user'].isin(users_to_remove) & data['user'].isin(user_set)].reset_index(drop=True)

    val_user_cleaned_set = clean_user(rest_data, val_user_set)
    test_user_cleaned_set = clean_user(rest_data, test_user_set)

    val_data = rest_data[rest_data["user"].isin(val_user_cleaned_set['user'])].reset_index(drop=True)
    test_data = rest_data[rest_data['user'].isin(test_user_cleaned_set['user'])].reset_index(drop=True)
    # print(train_data,val_data,test_data)

    # Split the test set with high ratings
    test_data_high_rating = test_data.loc[test_data['rating'] > 4].reset_index(drop=True)
    val_data_high_rating = val_data.loc[val_data['rating'] > 4].reset_index(drop=True)

    print("saving train_data.csv, val_data.csv, test_data.csv")
    all_data.to_csv('data/' + dataset + '/all_data.csv', index=False, sep='\t')
    train_data.to_csv('data/' + dataset + '/train_data.csv', index=False, sep='\t')
    val_data.to_csv('data/' + dataset + '/val_data.csv', index=False, sep='\t')
    test_data.to_csv('data/' + dataset + '/test_data.csv', index=False, sep='\t')

    print("saving train_list.txt, val_list.txt, test_list.txt")
    # train_list.txt
    train_data = train_data.sort_values(by=['user']).reset_index(drop=True)
    train_user_count = train_data.groupby('user')['user'].count()
    f_train = open('data/' + dataset + '/train_list.txt', 'w')
    u = train_data['user'][0]
    user_interaction_num = train_data.groupby('user')['user'].count()[u]
    f_train.write(str(u) + '\t' + str(user_interaction_num) + '\t' + str(train_data['item'][0]))
    for i in range(1, train_data.shape[0]):
        if train_data['user'][i] == train_data['user'][i - 1]:
            f_train.write('\t' + str(train_data['item'][i]))
        else:
            u = train_data['user'][i]
            user_interaction_num = train_user_count[u]
            f_train.write('\n' + str(u) + '\t' + str(user_interaction_num) + '\t' + str(train_data['item'][i]))
    f_train.close()

    def test_process(test_data, test_data_high_rating, val=True):
        # test_list.txt
        test_data = test_data.sort_values(by=['user']).reset_index(drop=True)
        # print("test_data:\n", test_data)
        if val:
            f_test = open('data/' + dataset + '/val_list.txt', 'w')
        else:
            f_test = open('data/' + dataset + '/test_list.txt', 'w')
        f_test.write(str(test_data['user'][0]) + '\t' + str(test_data['item'][0]))
        for i in range(1, test_data.shape[0]):
            if test_data['user'][i] == test_data['user'][i - 1]:
                f_test.write('\t' + str(test_data['item'][i]))
            else:
                f_test.write('\n' + str(test_data['user'][i]) + '\t' + str(test_data['item'][i]))
        f_test.close()

        # test_high_rating_list.txt
        test_data_high_rating = test_data_high_rating.sort_values(by=['user']).reset_index(drop=True)
        # print("test_data_high_rating:\n", test_data_high_rating)
        if val:
            f_test = open('data/' + dataset + '/val_high_rating_list.txt', 'w')
        else:
            f_test = open('data/' + dataset + '/test_high_rating_list.txt', 'w')
        f_test.write(str(test_data_high_rating['user'][0]) + '\t' + str(test_data_high_rating['item'][0]))
        for i in range(1, test_data_high_rating.shape[0]):
            if test_data_high_rating['user'][i] == test_data_high_rating['user'][i - 1]:
                f_test.write('\t' + str(test_data_high_rating['item'][i]))
            else:
                f_test.write(
                    '\n' + str(test_data_high_rating['user'][i]) + '\t' + str(test_data_high_rating['item'][i]))
        f_test.close()

        # test_list_rating.txt
        if val:
            f_test_rating = open('data/' + dataset + '/val_list_rating.txt', 'w')
        else:
            f_test_rating = open('data/' + dataset + '/test_list_rating.txt', 'w')
        f_test_rating.write(str(test_data['user'][0]) + '\t' + str(test_data['rating'][0]))
        for i in range(1, test_data.shape[0]):
            if test_data['user'][i] == test_data['user'][i - 1]:
                f_test_rating.write('\t' + str(test_data['rating'][i]))
            else:
                f_test_rating.write('\n' + str(test_data['user'][i]) + '\t' + str(test_data['rating'][i]))
        f_test_rating.close()

    test_process(val_data, val_data_high_rating, val=True)
    test_process(test_data, test_data_high_rating, val=False)

    # Record the number of interactions and historical interaction timestamps for each item
    print("saving item_interactions.csv")
    train_data = train_data.sort_values(by=['item', 'timestamp']).reset_index(drop=True)
    f_item_interaction = open('data/' + dataset + '/item_interactions.csv', 'w')
    item_id = train_data['item'][0]
    item_idx = 0
    interaction_num = 1
    str_to_write = str(int(train_data['timestamp'][0]))

    for i in range(1, train_data.shape[0]):
        if train_data['item'][i] == train_data['item'][i - 1]:
            interaction_num += 1
            str_to_write = str_to_write + ',' + str(int(train_data['timestamp'][i]))
        else:
            str_to_write = str(item_id) + ',' + str(interaction_num) + ',' + str_to_write + '\n'
            f_item_interaction.write(str_to_write)
            item_id = train_data['item'][i]
            item_idx += 1
            while item_id != item_idx:  # Items with no interactions
                str_to_write = str(item_idx) + ',' + str(0) + '\n'
                f_item_interaction.write(str_to_write)
                item_idx += 1
            str_to_write = str(int(train_data['timestamp'][i]))
            interaction_num = 1
    str_to_write = str(item_id) + ',' + str(interaction_num) + ',' + str_to_write + '\n'  # Last item
    f_item_interaction.write(str_to_write)
    item_idx += 1
    while item_num != item_idx:  # Items with no interactions exist
        str_to_write = str(item_idx) + ',' + str(0) + '\n'
        f_item_interaction.write(str_to_write)
        item_idx += 1
    f_item_interaction.close()
    print("saved!")


def PDA_test_pop(dataset):
    PDA_popularity = pd.read_csv('data/' + dataset + '/PDA_popularity.csv', sep='\t')
    PDA_popularity['9'] = PDA_popularity['8'] + opt.PDA_alpha * (PDA_popularity['8'] - PDA_popularity['7'])
    return PDA_popularity['9'].values
    '''PDA_popularity['9'] = (PDA_popularity['9'] - np.min(PDA_popularity['9'])) / \
                          (np.max(PDA_popularity['9']) - np.min(PDA_popularity['9']))
    PDA_popularity.to_csv('data/' + dataset + '/PDA_popularity.csv', index=0, sep='\t')'''
