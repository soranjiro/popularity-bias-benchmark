// Number of users: 5541, Number of items: 3568, Total interactions: 64706, density: 0.003272891118
#define item_num 3568
#define user_num 5541
const char *dataset_path = "./data/Amazon-Music/";

/*
<%
setup_pybind11(cfg)
%>
*/
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// #include<iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define _CRT_SECURE_NO_WARNINGS

namespace py = pybind11;

#include "common.cpp"
#include "common.h"

double *item_popularity[item_num];
double *item_rating[item_num];
double *item_ave_rating[item_num];
double *grad[item_num];
int *item_interaction[item_num]; //[itemid][[timestamp]]
int *user_interaction[user_num];

int idx, i, itr_num_item, j, itemid;
int timestamp;
double all_average_rating;
int interaction_num[item_num] = {0}; // すべてのアイテムのインタラクション数のみを保存する
int interaction_num_user[user_num] = {0};
double tau[item_num] = {10000000}; // 8000000;
double tmp = 0;

PYBIND11_MODULE(pybind_amazon_music, m)
{

  m.doc() = "pybind11 example module";

  define_pybind_functions(m);
}
