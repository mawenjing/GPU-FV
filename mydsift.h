#ifndef __DSIFT_H_
#define __DSIFT_H_

#include <cstdio>
#include <string>
#include <cctype>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace std;
using namespace cv;
/*
extern "C"{
	#include <vl/generic.h>
	#include <vl/stringop.h>
	#include <vl/pgm.h>
	#include <vl/sift.h>
	#include <vl/dsift.h>
	#include <vl/getopt_long.h>
};
*/
template <class T>
void printVec(vector<T> & tvec);

void dsift_init();
void dsift_finalize();
int dsift(char* imagePath, vector<vector<float> > &result);
int dsift_cpu(char* imagePath, float** siftresg, float* projection, float* projectionCenter);
//VlDsiftFilter * dsift(char* imagePath);
int dsift_gpu(char* imagePath, float** siftresg, float* projection, float* projectionCenter, float** siftframe);
//VlDsiftFilter * dsift(char* imagePath);
int dsift_train(char* imagePath, float** siftresg, float** siftframe, int height, int width);
#endif
