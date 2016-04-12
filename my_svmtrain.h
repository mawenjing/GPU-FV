#ifndef MY_SVMTRAIN_H
#define MY_SVMTRAIN_H
void train(float** enc, int* labels);
void do_predict(float** enc, int* labels, int* difficult);
void vlsvmtrain(double* vlW, double* vlb, float* indata, double* labels, int numData, float lambda);
double my_do_predict(float** encs, double* labels, int* difficult, double* vlW, double vlb, int class_index);
//double my_do_predict_acc(float** encs, double* labels, int* difficult, double* vlW, double vlb, int class_index);
double my_do_predict_acc(int class_index, int* preds, int* labels);
void calc_score(double** scores, float** encs, int* labels, double** vlW, double* vlb, int* preds);
#endif
