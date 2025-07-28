
#include "common.h"

void load_ratings_norm() {
  FILE *fp;

  char buffer[8192 * 16] = {0};
  char *line;

  // 初期化
  for (int i = 0; i < item_num; i++) {
    if (item_rating[i] != NULL) {
      free(item_rating[i]);
      item_rating[i] = NULL;
    }
    if (item_ave_rating[i] != NULL) {
      free(item_ave_rating[i]);
      item_ave_rating[i] = NULL;
    }

    item_rating[i] = (double *)malloc((interaction_num[i]) * sizeof(double));
    item_ave_rating[i] = (double *)malloc((interaction_num[i]) * sizeof(double));

    if (item_rating[i] == nullptr) {
      printf("Memory allocation failed for item_rating[%d]\n", i);
      fflush(stdout);
      exit(1);
    }
    if (item_ave_rating[i] == nullptr) {
      printf("Memory allocation failed for item_ave_rating[%d]\n", i);
      fflush(stdout);
      exit(1);
    }

    for (int j = 0; j < interaction_num[i]; j++) {
      item_rating[i][j] = -1;
      item_ave_rating[i][j] = -1;
    }
  }

  int all_review_counts = 0;
  // trainの評価値を読み込む
  if ((fp = fopen((std::string(dataset_path) + "all_data.csv").c_str(), "r")) != NULL) {
    bool first_line = true;
    while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
      if (first_line) {
        first_line = false;
        continue;
      }

      char *token = strtok(line, "\t");
      token = strtok(NULL, "\t");
      int item_id = atoi(token);

      token = strtok(NULL, "\t");
      // Normalize the rating by dividing by 5.0 to scale it between 0 and 1
      double rating = atof(token) / 5.0;

      token = strtok(NULL, "\t");

      token = strtok(NULL, "\t");
      int split_idx = atoi(token);

      if (split_idx == 9) {
        continue;
      }

      int s = 0;
      for (; s < interaction_num[item_id] && item_ave_rating[item_id][s] >= 0; s++) {
        // まだ評価値が入っていない場所を先頭探索
        // item_ave_ratingが0はそれまでの評価値が全て-1であることを示す
        // sの値はまだ評価値が入っていない場所のindex
      }

      item_rating[item_id][s] = rating;

      // ave_ratingを計算
      if (s == 0) {
        item_ave_rating[item_id][s] = 0;
      }
      else if (s < interaction_num[item_id]) {
        // item_ratings[item_id][0]からitem_ratings[item_id][s-1]までで-1でない個数を計算
        int count = 0;
        double tmp_ave_rating = 0;
        for (int k = 0; k < s; k++) {
          if (item_rating[item_id][k] > 0) {
            count++;
            tmp_ave_rating += item_rating[item_id][k];
          }
        }
        if (count != 0) {
          item_ave_rating[item_id][s] = tmp_ave_rating / count;
        } else {
          item_ave_rating[item_id][s] = 0;
        }

        if (item_ave_rating[item_id] == nullptr) {
          printf("Memory allocation failed for item_ave_rating[%d]\n", item_id);
          fflush(stdout);
          exit(1);
        }
      } else {
        printf("error in load_ratings. number of s is invalid\n");
        printf("item_id: %d, s: %d, interaction_num[item_id]: %d\n", item_id, s, interaction_num[item_id]);
        printf("s: %d, interaction_num[item_id]: %d\n", s, interaction_num[item_id]);
        fflush(stdout);
        exit(1);
      }

      if (rating > 0) {
        all_review_counts++;
        all_average_rating += rating;
      }
    }

    all_average_rating = all_average_rating / all_review_counts;

    // 評価値が入っていない場所に全体の平均評価値を入れる
    for (int item_id = 0; item_id < item_num; item_id++) {
      for (int s = 0; s < interaction_num[item_id]; s++) {
        if (item_ave_rating[item_id][s] <= 0) {
          item_ave_rating[item_id][s] = all_average_rating;

          // 0以下なのに，それまでに評価値が入っている場合はエラー
          if (s != 0 && item_ave_rating[item_id][s - 1] != all_average_rating) {
            printf("error in load_ratings norm. ratings in different point\n");
            printf("item_id: %d, s: %d, interaction_num[item_id]: %d\n", item_id, s, interaction_num[item_id]);
            printf("s: %d, interaction_num[item_id]: %d\n", s, interaction_num[item_id]);
            printf("item_ave_rating[item_id][s - 1]: %f\n", item_ave_rating[item_id][s - 1]);
            fflush(stdout);
            exit(1);
          }
        }
      }
    }

    printf("all_average_rating: %f\n", all_average_rating);
    fclose(fp);
  } else {
    printf("failed to open file! %s\n",
           (std::string(dataset_path) + "all_data.csv").c_str());
    fflush(stdout);
    exit(1);
  }

  printf("load_ratings_norm done\n");
}

void load_popularity(py::array_t<double> &tau_p) {
  FILE *fp;
  py::buffer_info buf1 = tau_p.request();
  auto tau_s = py::array_t<double>(buf1.size);
  double *ptr1 = (double *)buf1.ptr;

  char buffer[8192 * 16] = {0};
  char *line;

  if ((fp = fopen((std::string(dataset_path) + "item_interactions.csv").c_str(),
                  "r")) != NULL) {

    for (i = 0; i < item_num; i++) {
      line = fgets(buffer, sizeof(buffer), fp);
      idx = atoi(strtok(line, ","));
      tau[idx] = ptr1[idx];
      if (idx != i) {
        printf("%d", idx);
      }
      int itr_num_item = atoi(strtok(NULL, ","));
      interaction_num[i] = itr_num_item;
      if (item_interaction[i] != NULL) {
        free(item_interaction[i]);
        item_interaction[i] = NULL;
      }
      if (item_popularity[i] != NULL) {
        free(item_popularity[i]);
        item_popularity[i] = NULL;
      }
      if (grad[i] != NULL) {
        free(grad[i]);
        grad[i] = NULL;
      }

      item_interaction[i] = (int *)malloc((itr_num_item) * sizeof(int));
      item_popularity[i] = (double *)malloc((itr_num_item) * sizeof(double));
      grad[i] = (double *)malloc((itr_num_item) * sizeof(double));
      for (j = 0; j < itr_num_item; j++) {
        timestamp = atoi(strtok(NULL, ","));
        item_interaction[i][j] = timestamp;
      }
      item_popularity[i][0] = 0;
      grad[i][0] = 0;
      for (j = 1; j < itr_num_item; j++) {
        item_popularity[i][j] = (item_popularity[i][j - 1] + 1) * exp(-1 * double(item_interaction[i][j] - item_interaction[i][j - 1]) / tau[idx]);
        tmp = (item_interaction[i][j] - item_interaction[i][j - 1]) / tau[idx] / tau[idx];
        grad[i][j] = (grad[i][j - 1] + tmp) * exp(-1 * double(item_interaction[i][j] - item_interaction[i][j - 1]) / tau[idx]);
      }
    }

    fclose(fp);
  } else {
    printf("failed to open file! %s\n",
           (std::string(dataset_path) + "item_interactions.csv").c_str());
    fflush(stdout);
    exit(1);
  }
}

void load_popularity_count_review() {
  FILE *fp;

  char buffer[8192 * 16] = {0};
  char *line;

  if ((fp = fopen((std::string(dataset_path) + "item_interactions.csv").c_str(),
                  "r")) != NULL) {

    for (i = 0; i < item_num; i++) {
      line = fgets(buffer, sizeof(buffer), fp);
      idx = atoi(strtok(line, ","));
      if (idx != i) {
        printf("%d", idx);
      }
      int itr_num_item = atoi(strtok(NULL, ","));
      interaction_num[i] = itr_num_item;
      if (item_interaction[i] != NULL) {
        free(item_interaction[i]);
        item_interaction[i] = NULL;
      }
      if (item_popularity[i] != NULL) {
        free(item_popularity[i]);
        item_popularity[i] = NULL;
      }
      if (grad[i] != NULL) {
        free(grad[i]);
        grad[i] = NULL;
      }

      item_interaction[i] = (int *)malloc((itr_num_item) * sizeof(int));
      item_popularity[i] = (double *)malloc((itr_num_item) * sizeof(double));
      grad[i] = (double *)malloc((itr_num_item) * sizeof(double));
      for (j = 0; j < itr_num_item; j++) {
        timestamp = atoi(strtok(NULL, ","));
        item_interaction[i][j] = timestamp;
      }
      item_popularity[i][0] = 0;
      grad[i][0] = 0;
      for (j = 1; j < itr_num_item; j++) {
        item_popularity[i][j] = j;
        grad[i][j] = 1;
      }
    }

    fclose(fp);
  } else {
    printf("failed to open file! %s\n",
           (std::string(dataset_path) + "item_interactions.csv").c_str());
    fflush(stdout);
    exit(1);
  }
}

py::array_t<double> ari(py::array_t<double> &item_p,
                        py::array_t<double> &timestamp_p) {
  py::buffer_info buf1 = item_p.request();
  py::buffer_info buf2 = timestamp_p.request();

  auto ari_result = py::array_t<double>(buf1.size);
  py::buffer_info buf3 = ari_result.request();

  double *ptr1 = (double *)buf1.ptr;
  double *ptr2 = (double *)buf2.ptr;
  double *ptr3 = (double *)buf3.ptr;

  for (int i = 0; i < buf1.shape[0]; i++) {
    itemid = int(ptr1[i]);
    timestamp = int(ptr2[i]);
    if (interaction_num[itemid] == 0) {
      ptr3[i] = all_average_rating;
      continue;
    }
    j = 0;
    while (j + 100 < interaction_num[itemid] &&
           item_interaction[itemid][j + 100] <= timestamp) {
      j = j + 100;
    }
    while (j + 10 < interaction_num[itemid] &&
           item_interaction[itemid][j + 10] <= timestamp) {
      j = j + 10;
    }
    for (; item_interaction[itemid][j] <= timestamp &&
           j < interaction_num[itemid];
         j++) {
    }
    if (j == 0) {
      ptr3[i] = all_average_rating;
    } else if (item_interaction[itemid][j - 1] == timestamp) {
      ptr3[i] = item_ave_rating[itemid][j - 1];
    } else {
      ptr3[i] = item_ave_rating[itemid][j - 1];
    }
  }

  return ari_result;
}

py::array_t<double> ariall() {
  auto ari_result = py::array_t<double>(item_num);
  py::buffer_info buf1 = ari_result.request();

  double *ptr1 = (double *)buf1.ptr;

  for (int i = 0; i < item_num; i++) {
    if (interaction_num[i] == 0) {
      ptr1[i] = all_average_rating;
      continue;
    }
    ptr1[i] = item_ave_rating[itemid][interaction_num[itemid] - 1];
  }
  return ari_result;
}

py::array_t<double> popularity(py::array_t<double> &item_p,
                 py::array_t<double> &timestamp_p) {
  py::buffer_info buf1 = item_p.request();
  py::buffer_info buf2 = timestamp_p.request();

  if (buf1.size == 0 || buf2.size == 0) {
  throw std::runtime_error("Input arrays are empty.");
  }

  auto popularity_result = py::array_t<double>(buf1.size);
  py::buffer_info buf3 = popularity_result.request();

  double *ptr1 = (double *)buf1.ptr;
  double *ptr2 = (double *)buf2.ptr;
  double *ptr3 = (double *)buf3.ptr;

  for (int i = 0; i < buf1.shape[0]; i++) {
  int itemid = int(ptr1[i]);
  int timestamp = int(ptr2[i]);

  if (itemid < 0 || itemid >= item_num) {
    throw std::runtime_error("Invalid item ID: " + std::to_string(itemid));
  }

  if (interaction_num[itemid] == 0) {
    ptr3[i] = 0;
    continue;
  }

  int j = 0;
  while (j + 100 < interaction_num[itemid] &&
       item_interaction[itemid][j + 100] <= timestamp) {
    j = j + 100;
  }
  while (j + 10 < interaction_num[itemid] &&
       item_interaction[itemid][j + 10] <= timestamp) {
    j = j + 10;
  }
  for (; j < interaction_num[itemid] && item_interaction[itemid][j] <= timestamp; j++) {
  }

  if (j == 0) {
    ptr3[i] = 0;
  } else if (item_interaction[itemid][j - 1] == timestamp) {
    ptr3[i] = item_popularity[itemid][j - 1] + 1;
  } else {
    if (tau[itemid] <= 0) {
    throw std::runtime_error("Invalid tau value for item ID: " + std::to_string(itemid));
    }
    ptr3[i] = (item_popularity[itemid][j - 1] + 1) *
        exp(-1 * double(timestamp - item_interaction[itemid][j - 1]) /
          tau[itemid]);
  }
  }

  return popularity_result;
}

py::array_t<double> gradient(py::array_t<double> &item_p,
                             py::array_t<double> &timestamp_p) {
  py::buffer_info buf1 = item_p.request();
  py::buffer_info buf2 = timestamp_p.request();

  auto grad_result = py::array_t<double>(buf1.size);
  py::buffer_info buf4 = grad_result.request();

  double *ptr1 = (double *)buf1.ptr;
  double *ptr2 = (double *)buf2.ptr;
  double *ptr4 = (double *)buf4.ptr;

  for (int i = 0; i < buf1.shape[0]; i++) {
    itemid = int(ptr1[i]);
    timestamp = int(ptr2[i]);
    if (interaction_num[itemid] == 0) {
      ptr4[i] = 0;
      continue;
    }
    for (j = 0;
         item_interaction[itemid][j] < timestamp && j < interaction_num[itemid];
         j++) {
    }
    tmp = (timestamp - item_interaction[itemid][j - 1]) / tau[itemid] /
          tau[itemid];
    ptr4[i] = (grad[itemid][j - 1] + tmp) *
              exp(-1 * double(timestamp - item_interaction[itemid][j - 1]) /
                  tau[itemid]);
  }

  return grad_result;
}

void load_user_interation_val() {
  FILE *fp;
  char buffer[8192 * 128] = {0};
  char *line = 0;
  int user_id = 0;
  int itr_num_user, item_id;
  if ((fp = fopen((std::string(dataset_path) + "train_list.txt").c_str(),
                  "r")) != NULL) {
    while (!feof(fp)) {
      line = fgets(buffer, sizeof(buffer), fp);
      user_id = atoi(strtok(line, "\t"));
      itr_num_user = atoi(strtok(NULL, "\t"));
      interaction_num_user[user_id] = itr_num_user;
      user_interaction[user_id] = (int *)malloc((itr_num_user) * sizeof(int));
      for (j = 0; j < itr_num_user; j++) {
        item_id = atoi(strtok(NULL, "\t"));
        user_interaction[user_id][j] = item_id;
      }
    }
    fclose(fp);
  }
}

void load_user_interation_test() {
  FILE *fp;
  char buffer[8192 * 128] = {0};
  char *line = 0;
  int user_id = 0;
  int itr_num_user, item_id;
  if ((fp = fopen((std::string(dataset_path) + "train_list_t.txt").c_str(),
                  "r")) != NULL) {
    while (!feof(fp)) {
      line = fgets(buffer, sizeof(buffer), fp);
      user_id = atoi(strtok(line, "\t"));
      itr_num_user = atoi(strtok(NULL, "\t"));
      interaction_num_user[user_id] = itr_num_user;
      user_interaction[user_id] = (int *)malloc((itr_num_user) * sizeof(int));
      for (j = 0; j < itr_num_user; j++) {
        item_id = atoi(strtok(NULL, "\t"));
        user_interaction[user_id][j] = item_id;
      }
    }
    fclose(fp);
  }
}

py::array_t<double> negtive_sample(py::array_t<double> &user,
                                   py::array_t<double> &ng_num) {
  py::buffer_info buf1 = user.request();
  py::buffer_info buf2 = ng_num.request();

  auto neg_items = py::array_t<double>(buf1.size * 4);
  py::buffer_info buf3 = neg_items.request();

  double *ptr1 = (double *)buf1.ptr;
  double *ptr2 = (double *)buf2.ptr;
  double *ptr3 = (double *)buf3.ptr;

  int nega_num = int(ptr2[0]);

  int user_id, random_item;
  for (int i = 0; i < int(buf1.shape[0]); i++) {
    user_id = int(ptr1[i]);
    for (j = 0; j < nega_num;) {
      random_item = int(rand() % item_num);
      for (int n = 0; n < interaction_num_user[user_id]; n++) {
        if (random_item == user_interaction[user_id][n]) {
          random_item = int(rand() % item_num);
          n = 0;
          continue;
        }
      }
      ptr3[nega_num * i + j] = random_item;
      j++;
    }
  }
  return neg_items;
}
