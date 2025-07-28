
#ifndef COMMON_H
#define COMMON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// /*
// <%
// setup_pybind11(cfg)
// %>
// */
// #ifdef _MSC_VER
// #pragma warning(disable : 4996)
// #endif

// #include <stdio.h>
// #include <time.h>
// #include <math.h>
// #include <string.h>
// #include <stdlib.h>
// // #include<iostream>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// #define _CRT_SECURE_NO_WARNINGS

// namespace py = pybind11;

//! データセットごとの設定
// //* Ciao
// #define item_num 10724 // 64443  26813
// #define user_num 5868  // 75258  48799
// const char *dataset_path = "./data/Ciao/";

// //* amazon-music
// #define item_num 3568 // 64443  26813
// #define user_num 5541 // 75258  48799
// const char *dataset_path = "./data/Amazon-Music/";

// //* amazon-cds
// #define item_num 64443 // 64443  26813
// #define user_num 75258 // 75258  48799
// const char *dataset_path = "./data/Amazon-CDs_and_Vinyl/";

// //* douban-movie
// #define item_num 26813   //64443  26813
// #define user_num 48799   //75258  48799
// const char *dataset_path = "./data/Douban-movie/movie/";

//* douba-book
// #define item_num 64443   //64443  26813
// #define user_num 75258   //75258  48799
// const char *dataset_path = "./data/Douban-movie/book/";

// //* douban-music
// #define item_num 13779 // 64443  26813
// #define user_num 8876  // 75258  48799
// const char *dataset_path = "./data/Douban-movie/music/";

extern double *item_popularity[item_num]; // C_i
extern double *item_rating[item_num];
extern double *item_ave_rating[item_num]; // S_i
extern double *grad[item_num];
extern int *item_interaction[item_num]; //[itemid][[timestamp]]
extern int *user_interaction[user_num];

extern int idx, i, itr_num_item, j, itemid;
extern int timestamp;
extern double all_average_rating;
extern int interaction_num[item_num];
extern int interaction_num_user[user_num];
extern double tau[item_num];
extern double tmp;

// void load_ratings();
void load_ratings_norm();
void load_popularity(py::array_t<double> &tau_p);
void load_popularity_count_review();
// void load_rinc(py::array_t<double> &tau_p);
// void load_modec(py::array_t<double> &tau_p);
py::array_t<double> ari(py::array_t<double> &item_p, py::array_t<double> &timestamp_p);
py::array_t<double> ariall();
py::array_t<double> popularity(py::array_t<double> &item_p, py::array_t<double> &timestamp_p);
py::array_t<double> gradient(py::array_t<double> &item_p, py::array_t<double> &timestamp_p);
void load_user_interation_val();
void load_user_interation_test();
py::array_t<double> negtive_sample(py::array_t<double> &user, py::array_t<double> &ng_num);

void define_pybind_functions(pybind11::module &m)
{
//   m.def("load_ratings", &load_ratings);
  m.def("load_ratings_norm", &load_ratings_norm);
  m.def("load_popularity", &load_popularity);
  m.def("load_popularity_count_review", &load_popularity_count_review);
//   m.def("load_rinc", &load_rinc);
//   m.def("load_modec", &load_modec);
  m.def("ari", &ari);
  m.def("ariall", &ariall);
  m.def("popularity", &popularity);
  m.def("gradient", &gradient);
  m.def("load_user_interation_val", &load_user_interation_val);
  m.def("load_user_interation_test", &load_user_interation_test);
  m.def("negtive_sample", &negtive_sample);
  m.def("foo", []()
        { return "Hello, World!"; });
  m.def("foo2", []()
        { return "This is foo2!\n"; });
  m.def("add", [](int a, int b)
        { return a + b; });
  m.def("sub", [](int a, int b)
        { return a - b; });
  m.def("mul", [](int a, int b)
        { return a * b; });
  m.def("div", [](int a, int b)
        { return static_cast<float>(a) / b; });
}

#endif // COMMON_H
