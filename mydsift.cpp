#include <cstdio>
#include <string>
#include <cctype>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vector>
#include "dsift_org.h"
#include "mydsift.h"
//#include "/media/sdb1/mawenjing/yulei/vlfeat/vl/dsift.h"

using namespace std;
using namespace cv;
#include "cuda_files.h"


extern "C"{
#include "vl/generic.h"
#include "vl/stringop.h"
#include "vl/pgm.h"
  //#include "vl/dsift.h"
//#include "vl/sift.h"
#include "vl/getopt_long.h"
};

static int antialiasing;
#define MAX1 1200
template <class T>
void printVec(vector<T> & tvec)
{
  int len = tvec.size();
  int i=0;
  for(i=0; i<len; i++)
    {
      cout<<tvec[i];
      if(i!=len-1)
	cout<<",";
    }
  cout<<endl;
}
/*
function x = snorm(x)
	m=sum(x.^2,1);
n=sqrt(m);
x=bsxfun(@times, x, 1./max(1e-5, n));
*/
#define max(x, y) ((x)>(y)?(x):(y))
void normalize(const float* in, float* out, int numData) {
  for(int i=0;i<numData; i++) {
    float sum = 0.0;
    for(int j=0;j<128;j++) {
      
      sum += in[i*128+j] * in[i*128+j];
    }
    for(int j=0;j<128;j++) {
      out[i*128 + j] = in[i*128+j]/max(1e-5, sqrt(sum));
      //      cout<<i<<" "<<j<<" Normalize "<<out[i*128+j]<<endl;
    }
  }
}
float* siftframe;
#define DST_DIM 80
#define NUM_SCALES 8
float* dest;

#define MAXHEIGHT 600
#define MAXWIDTH 600
void dsift_init() {
  siftframe = (float*) malloc(MAXHEIGHT * MAXWIDTH* sizeof(float) * 9 * 2);  
}
void dsift_finalize() {
  free(siftframe);
  //  free(dest);
}

void new_resize_C1(cv::Mat& mtx, cv::Mat& outMat, double scale) {
  //      double scale4 = (double)(480.0/(double)height);
  int height = mtx.rows;
  int width = mtx.cols;
    cv::Mat mtxtmp(height, width, CV_32F);
    int i;

    /////////////////////Calculate the compare matrix
    int new_h = ceil(height*scale); //mtxtmp2.rows;
    int new_w = ceil(width*scale); //mtxtmp2.cols;
    int** indicesh=new int*[new_h];
    int** indicesw=new int*[new_w];
    double** weighth = new double*[new_h];
    double** weightw = new double*[new_w];
    double**mtxinter1 = new double*[new_h];
    double kernel_width = 2;
    if(scale<1 && antialiasing) {
      kernel_width = kernel_width/scale;
    }
    int P = ceil(kernel_width) + 2;

    for(i=0;i<new_h;i++) {
      indicesh[i] = new int[P];
      weighth[i] = new double[P];
      mtxinter1[i] = new double[width];
     }
    for(i=0;i<new_w;i++) {
      indicesw[i] = new int[P];
      weightw[i] = new double[P];
    }
    /*    int indicesw[MAX1][4];
    double weighth[MAX1][4];
    double weightw[MAX1][4];
    double mtxinter1[MAX1][MAX1];
    */

    //Contribution
    int left;
    double mat_u, inter1;

    double* weightsum = new double[new_h];
    for(i=0;i<new_h;i++)
      weightsum[i] = 0.0;

    for(i=0;i<new_h;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);
      for(int j=0;j<P;j++) {
	indicesh[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesh[i][j];
       
	//TODO: kernel_width/scale; inter1*scale, weighth * scale.
	if(antialiasing && scale < 1)
	  inter1 = inter1*scale;
	if(inter1>=-1 && inter1<0) {
	  weighth[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weighth[i][j] = 1-inter1;
	}
	else {
	  weighth[i][j] = 0;
	}
	if(antialiasing && scale < 1)
	  weighth[i][j] *= scale;
	//	cout<<j<<" weight "<<weightsum[j]<<", "<<weighth[i][j]<<endl;
	weightsum[i] += weighth[i][j];
      }
    }
    /*
    for(i=0;i<P;i++)
      cout<<"weight sum "<<weightsum[i]<<endl;
    */
    for(i=0;i<new_h;i++) {
      for(int j=0;j<P;j++) {
	if(weightsum[i]!=0)
	  weighth[i][j] = weighth[i][j]/weightsum[i];
      }
    }

    //Omitted the normalization
    for(i=0;i<new_h;i++) {
      for(int j=0;j<P;j++) {
	//	cout<<"Indices "<<indicesh[i][1]-1<<", "<<indicesh[i][2]-1<<endl;
	indicesh[i][j] = std::min(height, max(1, indicesh[i][j]));
      }
    }

    delete weightsum;
    weightsum = new double[new_w];
    ////////////////scale on width
    for(i=0;i<new_w;i++)
      weightsum[i] = 0.0;

    for(i=0;i<new_w;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);
      //      cout<<"Left is "<<kernel_width<<", "<<left<<endl;
      for(int j=0;j<P;j++) {
	indicesw[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesw[i][j];
	if(antialiasing && scale < 1)
	  inter1 = inter1*scale;

	if(inter1>=-1 && inter1<0) {
	  weightw[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weightw[i][j] = 1-inter1;
	}
	else {
	  weightw[i][j] = 0;
	}
	if(antialiasing && scale < 1)
	  weightw[i][j] *= scale;
	weightsum[i] += weightw[i][j];
      }
    }
    for(i=0;i<new_w;i++) {
      for(int j=0;j<P;j++) {
	weightw[i][j] = weightw[i][j]/weightsum[i];
      }
    }
    delete weightsum;
    //Omitted the normalization
    for(i=0;i<new_w;i++) {
      for(int j=0;j<P;j++)
	indicesw[i][j] = std::min(width, max(1, indicesw[i][j]));
    }

    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	mtxinter1[i][j] = 0.0;
	for(int k=1;k<P-1;k++) {
	  //	mtxinter1[i][j] = mtx.at<float>(indicesh[i][1]-1,j)*weighth[i][1] + mtx.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
	  mtxinter1[i][j] += mtx.at<float>(indicesh[i][k]-1, j)*weighth[i][k];
	}
	//      	cout<<i<<", "<<j<<"weight "<<weighth[i][1]<<", "<<weighth[i][2]<<" Inter "<<mtxinter1[i][j]<<endl;
      }
    }
    
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++) {
	outMat.at<float>(i,j) = 0.0;
	for(int k=1;k<P-1;k++) {
	  outMat.at<float>(i,j) += mtxinter1[i][indicesw[j][k]-1]*weightw[j][k];
	}
	//	outMat.at<float>(i,j) = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
      }
    }
    for(i=0;i<new_h;i++) {
      delete [] indicesh[i];
      delete[] weighth[i];
      delete[] mtxinter1[i];
    }
    for(i=0;i<new_w;i++) {
      delete[] indicesw[i];
      delete[] weightw[i];
    }
    delete[] indicesh;
    delete[] weighth;
    delete[] mtxinter1;
    delete[] indicesw;
    delete[] weightw;
}

void new_resize_C1_float(int height, int width, float* mtx, float* outMat, double scale, int antialiasing) { 
  float* mtxtmp= new float[height*width];
  int i;

  /////////////////////Calculate the compare matrix
    int new_h = ceil(height*scale); //mtxtmp2.rows;
    int new_w = ceil(width*scale); //mtxtmp2.cols;
    //    cout<<"News are "<<new_h<<", "<<new_w<<", "<<&outMat[new_h*new_w-1]<<endl;
    int** indicesh=new int*[new_h];
    int** indicesw=new int*[new_w];
    double** weighth = new double*[new_h];
    double** weightw = new double*[new_w];
    double**mtxinter1 = new double*[new_h];
    double kernel_width = 2;
    if(scale<1 && antialiasing) {
      kernel_width = kernel_width/scale;
    }
    int P = ceil(kernel_width) + 2;
    for(i=0;i<new_h;i++) {
      indicesh[i] = new int[P];
      weighth[i] = new double[P];
      mtxinter1[i] = new double[width];
    }
    for(i=0;i<new_w;i++) {
      indicesw[i] = new int[P];
      weightw[i] = new double[P];
    }

    /*    int indicesh[MAX1][4];
    int indicesw[MAX1][4];
    double weighth[MAX1][4];
    double weightw[MAX1][4];
    double mtxinter1[MAX1][MAX1];
    */

    //Contribution
    int left;
    double mat_u, inter1;

    double* weightsum = new double[new_h];
    for(i=0;i<new_h;i++)
      weightsum[i] = 0.0;

    for(i=0;i<new_h;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);

      for(int j=0;j<P;j++) {
	indicesh[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesh[i][j];
	if(antialiasing && scale < 1)
	  inter1 = inter1*scale;

	if(inter1>=-1 && inter1<0) {
	  weighth[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weighth[i][j] = 1-inter1;
	}
	else {
	  weighth[i][j] = 0;
	}
	if(antialiasing && scale < 1)
	  weighth[i][j] *= scale;
	weightsum[i] += weighth[i][j];
      }
    }

    for(i=0;i<new_h;i++) {
      for(int j=0;j<P;j++) {
	if(weightsum[i]!=0)
	  weighth[i][j] = weighth[i][j]/weightsum[i];
      }
    }

    //Omitted the normalization
    for(i=0;i<new_h;i++) {
      for(int j=0;j<P;j++) {
	//	cout<<"Indices "<<indicesh[i][1]-1<<", "<<indicesh[i][2]-1<<endl;
	indicesh[i][j] = std::min(height, max(1, indicesh[i][j]));
      }
    }
    delete weightsum;
    weightsum = new double[new_w];
    ////////////////scale on width
    for(i=0;i<new_w;i++)
      weightsum[i] = 0.0;

    for(i=0;i<new_w;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);
      for(int j=0;j<P;j++) {
	indicesw[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesw[i][j];
	if(antialiasing && scale < 1)
	  inter1 = inter1*scale;

	if(inter1>=-1 && inter1<0) {
	  weightw[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weightw[i][j] = 1-inter1;
	}
	else {
	  weightw[i][j] = 0;
	}
	if(antialiasing && scale < 1)
	  weightw[i][j] *= scale;
	weightsum[i] += weightw[i][j];
      }
    }
    for(i=0;i<new_w;i++) {
      for(int j=0;j<P;j++) {
	weightw[i][j] = weightw[i][j]/weightsum[i];
      }
    }
    delete weightsum;

    //Omitted the normalization
    for(i=0;i<new_w;i++) {
      for(int j=0;j<P;j++)
	indicesw[i][j] = std::min(width, max(1, indicesw[i][j]));
    }
    //////////////Now get the B G R values
    /*
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	//mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[0];
	mtxtmp[i*width + j] = (float)((float)(mtx[i*width*3 + j*3]) / 255.0);
      }
      }*/

    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	mtxinter1[i][j] = 0.0;
	for(int k=1;k<P-1;k++) {
	  //	mtxinter1[i][j] = mtx.at<float>(indicesh[i][1]-1,j)*weighth[i][1] + mtx.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
	  //WARNING: must have "-1"
	  mtxinter1[i][j] += mtx[(indicesh[i][k]-1)*width+ j]*weighth[i][k];
	}
	//	mtxinter1[i][j] = mtxtmp[(indicesh[i][1]-1) * width + j] * weighth[i][1] + mtxtmp[(indicesh[i][2]-1) * width + j]*weighth[i][2];
      }
    }

    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++) {
	//outMat.at<Vec3f>(i,j).val[0] = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
	//	cout<<"Indices "<<indicesw[j][1]<<", "<<indicesw[j][2]<<endl;
	outMat[i*new_w+j] = 0.0;
	for(int k=1;k<P-1;k++) {
	  outMat[i*new_w+j] += mtxinter1[i][indicesw[j][k]-1]*weightw[j][k];
	}
	//	outMat[i*new_w*3+j*3] = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
      }
    }
    for(i=0;i<new_h;i++) {
      delete [] indicesh[i];
      delete[] weighth[i];
      delete[] mtxinter1[i];
    }
    for(i=0;i<new_w;i++) {
      delete[] indicesw[i];
      delete[] weightw[i];
    }
    delete[] indicesh;
    delete[] weighth;
    delete[] mtxinter1;
    delete[] indicesw;
    delete[] weightw;

}
void new_resize_C3(cv::Mat& mtx, cv::Mat& outMat, double scale) {
  //      double scale4 = (double)(480.0/(double)height);
  int height = mtx.rows;
  int width = mtx.cols;
  //    cv::Mat mtxtmp2(ceil(height*scale), ceil(scale*width), CV_32FC3);
  //  cv::Mat mtxcompare(ceil(height*scale), ceil(scale*width), CV_32FC3);
    cv::Mat mtxtmp(height, width, CV_32F);
    int i;

    /////////////////////Calculate the compare matrix
    int new_h = ceil(height*scale); //mtxtmp2.rows;
    int new_w = ceil(width*scale); //mtxtmp2.cols;
    int** indicesh=new int*[new_h];
    int** indicesw=new int*[new_w];
    double** weighth = new double*[new_h];
    double** weightw = new double*[new_w];
    double**mtxinter1 = new double*[new_h];

    double kernel_width = 2;
    if(scale<1 && antialiasing) {
      kernel_width = kernel_width/scale;
    }
    int P = ceil(kernel_width) + 2;

    for(i=0;i<new_h;i++) {
      indicesh[i] = new int[P];
      weighth[i] = new double[P];
      mtxinter1[i] = new double[width];
    }
    for(i=0;i<new_w;i++) {
      indicesw[i] = new int[P];
      weightw[i] = new double[P];
    }

    /*    int indicesh[MAX1][4];
    int indicesw[MAX1][4];
    double weighth[MAX1][4];
    double weightw[MAX1][4];
    double mtxinter1[MAX1][MAX1];
    */
    //Contribution
    int left;
    double mat_u, inter1;

    double* weightsum = new double[new_h];
    for(i=0;i<new_h;i++)
      weightsum[i] = 0.0;

    for(i=0;i<new_h;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);

      for(int j=0;j<P;j++) {
	indicesh[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesh[i][j];
	if(antialiasing && scale < 1)
	  inter1 = inter1*scale;

	if(inter1>=-1 && inter1<0) {
	  weighth[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weighth[i][j] = 1-inter1;
	}
	else {
	  weighth[i][j] = 0;
	}
	if(antialiasing && scale < 1)
	  weighth[i][j] *= scale;
	weightsum[i] += weighth[i][j];
      }
    }
    for(i=0;i<new_h;i++) {
      for(int j=0;j<P;j++) {
	if(weightsum[i]!=0)
	  weighth[i][j] = weighth[i][j]/weightsum[i];
      }
    }

    for(i=0;i<new_h;i++) {
      for(int j=0;j<P;j++) {
	//	cout<<"Indices "<<indicesh[i][1]-1<<", "<<indicesh[i][2]-1<<endl;
	indicesh[i][j] = std::min(height, max(1, indicesh[i][j]));
      }
    }
    delete weightsum;
    weightsum = new double[new_w];

    ////////////////scale on width
    for(i=0;i<new_w;i++)
      weightsum[i] = 0.0;

    for(i=0;i<new_w;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);
      for(int j=0;j<P;j++) {
	indicesw[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesw[i][j];
	if(antialiasing && scale < 1)
	  inter1 = inter1*scale;

	if(inter1>=-1 && inter1<0) {
	  weightw[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weightw[i][j] = 1-inter1;
	}
	else {
	  weightw[i][j] = 0;
	}
	if(antialiasing && scale < 1)
	  weightw[i][j] *= scale;
	weightsum[i] += weightw[i][j];
      }
    }
    for(i=0;i<new_w;i++) {
      for(int j=0;j<P;j++) {
	weightw[i][j] = weightw[i][j]/weightsum[i];
      }
    }
    delete weightsum;

    //Omitted the normalization
    for(i=0;i<new_w;i++) {
      for(int j=0;j<P;j++)
	indicesw[i][j] = std::min(width, max(1, indicesw[i][j]));
    }
    //////////////Now get the B G R values
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[0];
      }
    }
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	mtxinter1[i][j] = 0.0;
	for(int k=1;k<P-1;k++)
	  mtxinter1[i][j] += mtxtmp.at<float>(indicesh[i][k]-1,j)*weighth[i][k];// + mtxtmp.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
      }
    }
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++) {
	outMat.at<Vec3f>(i,j).val[0] = 0.0;
	for(int k=1;k<P-1;k++)
	  outMat.at<Vec3f>(i,j).val[0] += mtxinter1[i][indicesw[j][k]-1]*weightw[j][k];// + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
      }
    }

    //////////////G
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[1];
      }
    }
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	mtxinter1[i][j] = 0.0;
	for(int k=1;k<P-1;k++)
	  mtxinter1[i][j] += mtxtmp.at<float>(indicesh[i][k]-1,j)*weighth[i][k];// + mtxtmp.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
      }
    }
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++) {
	outMat.at<Vec3f>(i,j).val[1] = 0.0;
	for(int k=1;k<P-1;k++)
	  outMat.at<Vec3f>(i,j).val[1] += mtxinter1[i][indicesw[j][k]-1]*weightw[j][k];// + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
      }
    }

    ////////////////R
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[2];
      }
    }
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	mtxinter1[i][j] = 0.0;
	for(int k=1;k<P-1;k++)
	  mtxinter1[i][j] += mtxtmp.at<float>(indicesh[i][k]-1,j)*weighth[i][k];// + mtxtmp.at<float>
      }
    }
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++) {
	outMat.at<Vec3f>(i,j).val[2] = 0.0;
	for(int k=1;k<P-1;k++)
	  outMat.at<Vec3f>(i,j).val[2] += mtxinter1[i][indicesw[j][k]-1]*weightw[j][k];// + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
      }
    }
    for(i=0;i<new_h;i++) {
      delete [] indicesh[i];
      delete[] weighth[i];
      delete[] mtxinter1[i];
    }
    for(i=0;i<new_w;i++) {
      delete[] indicesw[i];
      delete[] weightw[i];
    }
    delete[] indicesh;
    delete[] weighth;
    delete[] mtxinter1;
    delete[] indicesw;
    delete[] weightw;

}

void new_resize_C3_float(int height, int width, uchar* mtx, float* outMat, double scale) { //cv::Mat& mtx, cv::Mat& outMat, double scale) {
  /*  int height = mtx.rows;
  int width = mtx.cols;
  */
  //  cv::Mat mtxtmp(height, width, CV_32F);
  //  cout<<"Size "<<height*width;
  float* mtxtmp= new float[height*width];
  int i;

  /////////////////////Calculate the compare matrix
    int new_h = ceil(height*scale); //mtxtmp2.rows;
    int new_w = ceil(width*scale); //mtxtmp2.cols;
    //    cout<<"News are "<<new_h<<", "<<new_w<<", "<<&outMat[new_h*new_w-1]<<endl;
    int** indicesh=new int*[new_h];
    int** indicesw=new int*[new_w];
    double** weighth = new double*[new_h];
    double** weightw = new double*[new_w];
    double**mtxinter1 = new double*[new_h];

    for(i=0;i<new_h;i++) {
      indicesh[i] = new int[4];
      weighth[i] = new double[4];
      mtxinter1[i] = new double[width];
    }
    for(i=0;i<new_w;i++) {
      indicesw[i] = new int[4];
      weightw[i] = new double[4];
    }

    /*    int indicesh[MAX1][4];
    int indicesw[MAX1][4];
    double weighth[MAX1][4];
    double weightw[MAX1][4];
    double mtxinter1[MAX1][MAX1];
    */

    int kernel_width = 2;
    //Contribution
    int left;
    double mat_u, inter1;
    int P = ceil(kernel_width) + 2;

    for(i=0;i<new_h;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);

      for(int j=0;j<4;j++) {
	indicesh[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesh[i][j];
	if(inter1>=-1 && inter1<0) {
	  weighth[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weighth[i][j] = 1-inter1;
	}
	else {
	  weighth[i][j] = 0;
	}
      }
    }
    //Omitted the normalization
    for(i=0;i<new_h;i++) {
      for(int j=0;j<4;j++) {
	//	cout<<"Indices "<<indicesh[i][1]-1<<", "<<indicesh[i][2]-1<<endl;
	indicesh[i][j] = std::min(height, max(1, indicesh[i][j]));
      }
    }

    ////////////////scale on width
    for(i=0;i<new_w;i++) {
      mat_u = (double)(i+1)/scale + 0.5*(1-1/scale);
      left = floor(mat_u-kernel_width/2);
      for(int j=0;j<4;j++) {
	indicesw[i][j] = left + j;//bsxfun(@plus, left, 0:P-1); TODO
	inter1 = mat_u - indicesw[i][j];
	if(inter1>=-1 && inter1<0) {
	  weightw[i][j] = 1+inter1;
	}
	else if(inter1>=0 && inter1<=1) {
	  weightw[i][j] = 1-inter1;
	}
	else {
	  weightw[i][j] = 0;
	}
      }
    }
    //Omitted the normalization
    for(i=0;i<new_w;i++) {
      for(int j=0;j<4;j++)
	indicesw[i][j] = std::min(width, max(1, indicesw[i][j]));
    }
    //////////////Now get the B G R values
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	//mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[0];
	mtxtmp[i*width + j] = (float)((float)(mtx[i*width*3 + j*3]) / 255.0);
      }
    }

    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	//	mtxinter1[i][j] = mtxtmp.at<float>(indicesh[i][1]-1,j)*weighth[i][1] + mtxtmp.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
	mtxinter1[i][j] = mtxtmp[(indicesh[i][1]-1) * width + j] * weighth[i][1] + mtxtmp[(indicesh[i][2]-1) * width + j]*weighth[i][2];
      }
    }

    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++) {
	//outMat.at<Vec3f>(i,j).val[0] = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
	//	cout<<"Indices "<<indicesw[j][1]<<", "<<indicesw[j][2]<<endl;
	outMat[i*new_w*3+j*3] = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
      }
    }

#if 1
    //////////////G
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	//	mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[1];
	mtxtmp[i*width + j] = (float)((float)(mtx[i*width*3 + j*3 + 1]) / 255.0);;
      }
    }
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	//	mtxinter1[i][j] = mtxtmp.at<float>(indicesh[i][1]-1,j)*weighth[i][1] + mtxtmp.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
	mtxinter1[i][j] = mtxtmp[(indicesh[i][1]-1) * width + j] * weighth[i][1] + mtxtmp[(indicesh[i][2]-1) * width + j]*weighth[i][2];
      }
    }
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++)
	outMat[i*new_w*3+j*3+1] = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
    }

    ////////////////R
    for(i=0;i<height;i++) {
      for(int j=0;j<width;j++) {
	//	mtxtmp.at<float>(i,j)=mtx.at<Vec3f>(i,j).val[2];
	mtxtmp[i*width + j] = (float)((float)(mtx[i*width*3 + j*3 + 2]) / 255.0);//mtx[i*width*3 + j*3 + 2];
      }
    }
    for(i=0;i<new_h;i++) {
      for(int j=0;j<width;j++) {
	//mtxinter1[i][j] = mtxtmp.at<float>(indicesh[i][1]-1,j)*weighth[i][1] + mtxtmp.at<float>(indicesh[i][2]-1, j)*weighth[i][2];
	mtxinter1[i][j] = mtxtmp[(indicesh[i][1]-1) * width + j] * weighth[i][1] + mtxtmp[(indicesh[i][2]-1) * width + j]*weighth[i][2];
      }
    }
    //We only need weight[2],[3], indices[2],[3]
    for(i=0;i<new_h;i++) {
      for(int j=0;j<new_w;j++)
	outMat[i*new_w*3+j*3+2] = mtxinter1[i][indicesw[j][1]-1]*weightw[j][1] + mtxinter1[i][indicesw[j][2]-1]*weightw[j][2];
    }
#endif
    for(i=0;i<new_h;i++) {
      delete [] indicesh[i];
      delete[] weighth[i];
      delete[] mtxinter1[i];
    }
    for(i=0;i<new_w;i++) {
      delete[] indicesw[i];
      delete[] weightw[i];
    }
    delete[] indicesh;
    delete[] weighth;
    delete[] mtxinter1;
    delete[] indicesw;
    delete[] weightw;
    //    cout<<"mtxtmp "<<mtxtmp<<", "<<indicesh<<", "<<weighth<<", "<<mtxinter1<<", "<<indicesw<<", "<<weightw<<", "<<outMat<<", "<<mtx<<endl;
    delete mtxtmp;
    //    cout<<"All finished"<<endl;
}

int dsift_cpu(char* imagePath, float** siftresg, float* projection, float* projectionCenter) { //, char* imindex, int height, int width){
  //char *imagePath = image;
  double start, finish;
  float duration;
  //  int height, width;

  //TODO: set antialiasing
  antialiasing = 1;

  int i, noctaves=4, nlevels=2, o_min=0;
  ///////////////TEST//////////////
  //  IplImage *Image = cvLoadImage(imagePath, 0);
  //char imagename[256], imagesizename[256];
  //  sprintf(imagename, "rgbs%s", imindex);
  /*
  sprintf(imagesizename, "imagesize%s", imindex);
  ifstream isize(imagesizename);
  isize>>height;
  isize>>width;
  isize.close();
  */
  start = wallclock();
  //  IplImage *Image = cvLoadImage(imagePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  //  vl_sift_pix *ImageData = new vl_sift_pix[Image->height * Image->width];
  Mat Image = imread(imagePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  unsigned char Pixel;

  int height = Image.rows;
  int width = Image.cols;
  cout<<"Image "<<imagePath<<", "<<height<<", "<<width<<endl;

  float scales[NUM_SCALES] = { 1.414213562373095, 1, 0.707106781186548, 0.5, 0.353553390593274, 0.25, 0.176776695296637, 0.125};
  //float *siftres = (float*)malloc(sizeof(float)*height * width*128);
  cv::Mat mtx(height, width, CV_32FC3); 
  int pre_size = 0;
  for(int scale=0;scale<NUM_SCALES;scale++) {
    //    cv::Mat mtx_out(ceil((height)*scales[scale]), ceil(scales[scale]*(width)), CV_32F); 
    int newh = ceil(height*scales[scale]);
    int neww = ceil(width*scales[scale]);
    VlDsiftFilter* dSiftFilt = vl_dsift_new(neww, newh);
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;
    if(framex<=0 || framey<=0 || sizex<=0 || sizey<=0) {
      vl_dsift_delete(dSiftFilt);
      break;
    }

    pre_size += sizey*sizex;
    //WARNING: must delete it!!
    vl_dsift_delete(dSiftFilt);
  }
  int total_size = pre_size;
  siftresg[0] = (float*) malloc(sizeof(float)*total_size*128);

  //  siftresg[0] = (float*) malloc(height * width* sizeof(float) * 128 * 9);
  float* cur_res;

  //  float scales[NUM_SCALES] = {1, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125};

  //  mtx=imread(imagePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

  for(i=0; i<height; i++)
    {
      for(int j=0; j<width; j++)
	{
	  Pixel = Image.at<unsigned char>(i, j*3); //(unsigned char*)(Image->imageData + (i* Image->width + j)*3);
	  int p1=(int)Pixel;
	  mtx.at<Vec3f>(i,j).val[0]=(float)((int)(Pixel)/255.0);
	  //	  cout<<"Values "<<(int)Pixel<<" ";
	  Pixel = Image.at<unsigned char>(i, j*3+1);
	  int p2 = (int)Pixel;
	  mtx.at<Vec3f>(i,j).val[1]=(float)((int)(Pixel)/255.0);
	  //	  cout<<"Values "<<(int)Pixel<<" ";
	  Pixel = Image.at<unsigned char>(i, j*3+2);
	  mtx.at<Vec3f>(i,j).val[2]=(float)((int)(Pixel)/255.0);
	  //      	  if((int)Pixel!=p1 || p1!=p2)
	  //	    cout<<i<<" "<<j<<" Values "<<(int)Pixel<<endl;
	  //	  cout<<"Image data "<<mtx.at<Vec3f>(i,j).val[0]<<", "<<mtx.at<Vec3f>(i,j).val[1]<<", "<<mtx.at<Vec3f>(i,j).val[2]<<endl;
	}
    }
  //  cout<<"Read OK "<<wallclock() - start<<endl;
  //WARNING: resize if height is bigger than 480
  start  = wallclock();
  
  if(height>480) {
    double scale4 = (double)(480.0/(double)height);
    /*
    cv::Mat mtx1(ceil(height*scale4), ceil(scale4*width), CV_32FC3);
    cv::resize(mtx, mtx1, mtx1.size(), scale4, scale4, 1) ;//INTER_AREA);
    for(int k=0;k<mtx1.rows;k++) {
      for (int j=0;j<mtx1.cols; j++) {
	for(int x=0;x<3;x++) {
	  mtx1.at<Vec3f>(k, j).val[x] = std::min(1.0, max(mtx1.at<Vec3f>(k, j).val[x], 0.0));
	}
      }
    }
    */
    cv::Mat mtx1(ceil(height*scale4), ceil(scale4*width), CV_32FC3);
    new_resize_C3(mtx, mtx1, scale4);

    mtx = mtx1;
    height = mtx1.rows;
    width = mtx1.cols;
  }
  //rgb2gray
  cv::Mat mtx_in(height, width, CV_32F);
  //  float coef[3] = {0.2989, 0.5870, 0.1140};
  //  float coef[3] = {0.1140, 0.5870, 0.2989};
  float coef[3] = {0.114020904255103, 0.587043074451121, 0.298936021293776};
  for(i=0;i<height;i++) {
    for(int j=0;j<width;j++) {
      //      mtx_in.at<float>(i,j) = 0.0;

      if(mtx.at<Vec3f>(i,j).val[0] == mtx.at<Vec3f>(i,j).val[1] && mtx.at<Vec3f>(i,j).val[0] == mtx.at<Vec3f>(i,j).val[2])
	mtx_in.at<float>(i,j) = mtx.at<Vec3f>(i,j).val[0];
      else {

	float tmp = 0.0;
	for(int k=0;k<3;k++) {
	  //	mtx_in.at<float>(i,j) += mtx.at<Vec3f>(i,j).val[k]*coef[k];
	  tmp += mtx.at<Vec3f>(i,j).val[k]*coef[k]; //(double)mtx.at<Vec3f>(i,j).val[k] * coef[k];
	}
	mtx_in.at<float>(i,j) = tmp;
      }
    }
  }
  //  cout<<"Resize OK "<<wallclock() - start<<endl;
  //Size of center is 80*128 
  i=0;

  pre_size = 0;
  //  int width = Image->width;
  //  int height = Image->height;
  start = wallclock();
  for(int scale=0;scale<NUM_SCALES;scale++) {
    cv::Mat mtx_out(ceil((height)*scales[scale]), ceil(scales[scale]*(width)), CV_32F); 
    int newh = mtx_out.rows;
    int neww = mtx_out.cols;
    cout<<"New "<<newh<<", "<<neww<<". "<<scales[scale]<<endl;
    //    start = wallclock();    
    /*    if(scale<=2) {
      cv::resize(mtx_in, mtx_out, mtx_out.size(), scales[scale], scales[scale],INTER_CUBIC);
    }
    else {
    */
    //cv::resize(mtx_in, mtx_out, mtx_out.size(), scales[scale], scales[scale],1);
    for(int i=0;i<mtx_in.rows;i++) {
      for(int j=0;j<mtx_in.cols;j++) {
	//	cout<<"Before resize "<<i<<", "<<j<<", "<<mtx_in.at<float>(i,j)<<endl;
	//	printf("Be %.8f\n", mtx_in.at<float>(i,j));
      }
    }

    new_resize_C1(mtx_in, mtx_out, scales[scale]);
      //}


    //1 is inter-linear, 2 is inter-cubic
    cv::Mat new_out(neww, newh, CV_32F);
    cv::transpose(mtx_out, new_out);
    cur_res = (float*) new_out.data;    

    i=0;
    /*
    float* tmp = new float[neww*newh];
    char name[256];
    sprintf(name, "feature%d", scale+1);
    ifstream code(name);
    while(code>> tmp[i++]) {
    }
    code.close();

    for(int j=0;j<newh;j++)
      for(int k=0;k<neww;k++)
	cur_res[j*neww+k] = tmp[j+newh*k];
    */
    VlDsiftFilter* dSiftFilt = vl_dsift_new(new_out.cols, new_out.rows);



    /*
    for(int k=0;k<newh;k++) {
      for(int j=0;j<neww;j++) {
       	cout<<"B "<<k<<", "<<j<<", "<<cur_res[k*neww+j]<<endl;
      }
      }*/
    

    i=0;
    /*
    if(scale == 0) {
    ifstream f1("feature1");        
    cout<<"New is "<<neww<<", "<<newh<<endl;
    for(int i=0;i<neww*newh;i++) {
      float temp;
      f1>>temp;
      
      if(abs(temp - cur_res[i])>0.00000001)
	printf("%d CPU %.9f, should be %.8f\n", i, cur_res[i], temp);
      
      //      cout<<i<<" CPU "<<cur_res[i]<<endl;
    }
    }
    */
    sift_time = 0;
    kernel_time = 0;
    copy_time = 0;
    filt_time = 0;
    wait_time = 0;
    init_time = 0;
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;

    if(sizex<=0 || sizey<=0) {
      vl_dsift_delete(dSiftFilt);
      break;
    }
    double    start1 = wallclock();
    my_vl_dsift_process(dSiftFilt, cur_res, scales[scale], 0);
    //    vl_dsift_process(dSiftFilt, cur_res);
    
    int cur_size = pre_size * 128;

    const float* x = vl_dsift_get_descriptors(dSiftFilt);
    //    normalize(x, siftresg[0]+cur_size, vl_dsift_get_keypoint_num(dSiftFilt))

    /*
    //TRANSPOSE again
    for(int j=0;j<sizey;j++) {
      for(int k=0;k<sizex;k++) {
	//siftresg[0][cur_size + k*sizey+j] = temp[j*sizex + k];
	for(int m=0;m<128;m++)
	  temp[128*(k*sizey + j)+m ] = x[128*(j*sizex + k)+m];
      }
    }    
    */
    cout<<"It is OK"<<endl;
    normalize(x, siftresg[0]+cur_size, vl_dsift_get_keypoint_num(dSiftFilt));

    //TRANSPOSE
    
    //    float* temp = (float*) malloc(sizex*sizey*sizeof(float)*128);
    /*
    memcpy(temp, siftresg[0]+cur_size, sizex*sizey*128*sizeof(float));
    for(int j=0;j<sizey;j++) {
      for(int k=0;k<sizex;k++) {
	for(int m=0;m<128;m++) {
	  siftresg[0][cur_size+(j*sizex+k)*128+m] = temp[(k*sizey+j)*128+m];
	}
      }
    }
    */
    //    sift_time += wallclock() - start;
    cout<<"Sift time "<<wallclock() - start1<<endl;
    /*
    for(i=0;i<sizex*sizey*128;i++) {
      siftresg[0][cur_size+i] = x[i];
    }
    */
    //ADD FRAME HERE
    for(i=0;i<sizey*sizex;i++) {
      siftframe[pre_size*2 + i*2] = dSiftFilt->frames[i].y;
      siftframe[pre_size*2 + i*2+1] = dSiftFilt->frames[i].x;
    }
    pre_size += sizey*sizex;
    if(scale == 0) {
      /*
  ifstream f("output");
  int numData = sizey*sizex;
  cout<<"OK? "<<endl;
  float* tmp = siftresg[0]+cur_size;
  for(int i=0;i<numData*128;i++) //numData*82-100;i<numData*82;i++)
    {
      float temp;
      f>>temp;
      if(abs(temp-tmp[i])>0.00001)
	cout<<i/128<<", "<<i%128<<" Tmp 2 "<<tmp[i]<<", should "<<temp<<endl;
    }  
  f.close();
      */
    }

    /*    
    for(i=0;i<sizex*sizey;i++) {
      for(int j=0;j<128;j++)
	cout<<scale<<", "<<i<<", "<<j<<" Before projection "<<x[i]<<", "<<siftresg[0][i*128+j+cur_size]<<endl;
	}*/
    
    //    free(temp);
#if 0
    cout<<"Scale "<<scale<<endl;
    for(int dim=cur_size;dim<pre_size;dim++) {    
      if(scale>0 && abs(siftres[dim] - siftresg[0][dim]) > 0.001) 
	{
	  cout<<"Scale "<<scale<<", at "<<dim<<" right "<<siftres[dim]<<", my "<<siftresg[0][dim]<<endl;
	} 
    }
#endif
    vl_dsift_delete(dSiftFilt);
  }
  //siftresg[0] has the descriptors now
  //Do PCA  
  //  cout<<"process time "<<wallclock() - start<<endl;
  float* dest=(float*)malloc(pre_size*(DST_DIM+2)*sizeof(float));
  
  start = wallclock();
  //  gpu_pca_mm(projection, projectionCenter, siftresg[0], dest, pre_size, DST_DIM);
  
  for(i=0;i<DST_DIM;i++) {
    for(int j=0;j<pre_size;j++) {
      double sum = 0;
      for(int k=0;k<128;k++) {
	//	if(j==2161)
	//	  cout<<"Sift resg "<<siftresg[0][k+j*128]<<endl;
	sum += projection[i+80*k]* (siftresg[0][k+j*128] - projectionCenter[k]);
      }
      dest[i+j*(2+DST_DIM)] = sum;
      //      dest[i*pre_size+j]=sum;
      //      cout<<"Dest "<<dest[i*pre_size+j]<<endl;
    }
  }
  
  
  for(i=0;i<pre_size;i++) {
    //TODO: optimize the layout
    float halfwidth = ((float)width)/2;
    float halfheight = ((float)height)/2;
    dest[i*(DST_DIM+2) +DST_DIM] = (siftframe[i*2]-halfwidth)/width;
    dest[i*(DST_DIM+2) + DST_DIM+1] = (siftframe[i*2+1]-halfheight)/height;
    //  dest[pre_size*DST_DIM+i] = (siftframe[i*2]-width/2)/width;
    // dest[pre_size*(DST_DIM+1)+i] = (siftframe[i*2]-width/2)/width;
  }
  /*  
  ifstream code("feature");
  int numData = pre_size;
  float* temp = (float*)malloc(82*numData*sizeof(float));
  i=0;
  while(code>>temp[i++]) {
  }
  code.close();

  for(i=0;i<numData*82;i++) {
    if( abs(temp[i] - dest[i]) > 0.00001) 
      {
      cout<<i/82<<" "<<i%82<<" Tmp 2 "<<dest[i]<<", should be "<<temp[i]<<endl;
      //      dest[i] = temp[i];
      }
      }
  free(temp);
  */

  /*
  for(i=0;i<pre_size;i++) {
    for(int j=0;j<DST_DIM;j++) {
      if(abs(dest_g[i*82+j]-dest[i*82+j])>0.01) 
      {
	cout<<"Wrong "<<i<<", "<<j<<", "<<dest_g[i*82+j]<<", should "<<dest[i*82+j]<<endl;
      }
    }
  }
  */
  //  delete[] dest_g;
  //////////////////
  //  sift_time += wallclock() - start;
  float* temp1 = siftresg[0];
  siftresg[0] = dest;
  free(temp1);
  int terms  = pre_size;
  //  cout<<"Terms "<<pre_size<<endl;
  return terms;
}

int dsift_train(char* imagePath, float** siftresg, float** siftframe, int height, int width){
  //char *imagePath = image;
  double start, finish;
  float duration;
  int i, dimension = 128;
  Mat Image = imread(imagePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

  unsigned char Pixel;
  height = Image.rows;
  width = Image.cols;

  //  cout<<"Image "<<imagePath<<", "<<height<<", "<<width<<endl;
  float* cur_res;// = (float*) malloc(height*width*sizeof(float)*128*4);
  float scales[NUM_SCALES] = { 1.414213562373095, 1, 0.707106781186548, 0.5, 0.353553390593274, 0.25, 0.176776695296637, 0.125};

  cv::Mat mtx(height, width, CV_32FC3); 

  for(i=0; i<height; i++)
    {
      for(int j=0; j<width; j++)
	{
	  Pixel = Image.at<unsigned char>(i, j*3); 
	  mtx.at<Vec3f>(i,j).val[0]=(float)((float)(Pixel)/255.0);
	  Pixel = Image.at<unsigned char>(i, j*3+1);
	  mtx.at<Vec3f>(i,j).val[1]=(float)((float)(Pixel)/255.0);
	  Pixel = Image.at<unsigned char>(i, j*3+2);
	  mtx.at<Vec3f>(i,j).val[2]=(float)((float)(Pixel)/255.0);
	}
    }
  //WARNING: resize if height is bigger than 480
  
  if(height>480) {
    double scale4 = (double)(480.0/(double)height);
    cv::Mat mtx1(ceil(height*scale4), ceil(scale4*width), CV_32FC3);
    new_resize_C3(mtx, mtx1, scale4);
    mtx = mtx1;
    height = mtx1.rows;
    width = mtx1.cols;
  }
  
  i=0;

  //rgb2gray
  cv::Mat mtx_in(height, width, CV_32F);
  //  float coef[3] = {0.2989, 0.5870, 0.1140};
  float coef[3] = {0.114020904255103, 0.587043074451121, 0.298936021293776};
  for(i=0;i<height;i++) {
    for(int j=0;j<width;j++) {
      if(mtx.at<Vec3f>(i,j).val[0] == mtx.at<Vec3f>(i,j).val[1] && mtx.at<Vec3f>(i,j).val[0] == mtx.at<Vec3f>(i,j).val[2])
	mtx_in.at<float>(i,j) = mtx.at<Vec3f>(i,j).val[0];
      else {

	float tmp = 0.0;
	for(int k=0;k<3;k++) {
	  //	mtx_in.at<float>(i,j) += mtx.at<Vec3f>(i,j).val[k]*coef[k];
	  tmp += mtx.at<Vec3f>(i,j).val[k]*coef[k]; //(double)mtx.at<Vec3f>(i,j).val[k] * coef[k];
	}
	mtx_in.at<float>(i,j) = tmp;
      }
    }
  }

  int pre_size = 0;
  //  int width = Image->width;
  //  int height = Image->height;
  for(int scale=0;scale<NUM_SCALES;scale++) {
    //    cv::Mat mtx_out(ceil((height)*scales[scale]), ceil(scales[scale]*(width)), CV_32F); 
    int newh = ceil(height*scales[scale]);
    int neww = ceil(width*scales[scale]);
    VlDsiftFilter* dSiftFilt = vl_dsift_new(neww, newh);
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;
    if(framex<=0 || framey<=0 || sizex < 0 || sizey<0) {
      vl_dsift_delete(dSiftFilt);
      break;
    }
    pre_size += sizey*sizex;
    //WARNING: must delete it!!
    vl_dsift_delete(dSiftFilt);
  }
  int total_size = pre_size;
  pre_size = 0;

  siftresg[0] = (float*) malloc(total_size * sizeof(float) * 128);
  siftframe[0] = (float*) malloc(total_size* sizeof(float) * 2);

  for(int scale=0;scale<NUM_SCALES;scale++) {
    cv::Mat mtx_out(ceil((height)*scales[scale]), ceil(scales[scale]*(width)), CV_32F);
    int newh = mtx_out.rows;
    int neww = mtx_out.cols;
    double start3 = wallclock();
    
    new_resize_C1(mtx_in, mtx_out, scales[scale]);

    //    cout<<scales[scale]<<": "<<newh<<", "<<neww<<": Resize time "<<wallclock() - start3<<endl;

    VlDsiftFilter* dSiftFilt = vl_dsift_new(mtx_out.cols, mtx_out.rows);

    cur_res = (float*) mtx_out.data;
    start = wallclock();
    sift_time = 0;
    kernel_time = 0;
    copy_time = 0;
    filt_time = 0;
    wait_time = 0;
    init_time = 0;
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;


    int cur_size = pre_size * dimension;
    if(framex<=0 || framey<=0 || sizex<=0 || sizey<=0) {
      //      cuda_clean();
      vl_dsift_delete(dSiftFilt);
      break;
    }

    /////////////////////////////

    //  my_vl_dsift_process_gpu(dSiftFilt, siftres, siftresg[0]);
    //    my_vl_dsift_process(dSiftFilt, cur_res, scales[scale], 1);
    my_vl_dsift_process(dSiftFilt, cur_res, scales[scale], 1);

    const float* x = vl_dsift_get_descriptors(dSiftFilt);

    for(i=0;i<sizex*sizey*128;i++) {
      siftresg[0][cur_size+i] = x[i];
      //  cout<<"Data "<<x[i]<<", "<<siftresg[0][cur_size+i]<<endl;
      }
    //ADD FRAME HERE
    for(i=0;i<sizey*sizex;i++) {
      siftframe[0][pre_size*2 + i*2] = dSiftFilt->frames[i].x;
      siftframe[0][pre_size*2 + i*2+1] = dSiftFilt->frames[i].y;
    }
    pre_size += sizey*sizex;
    vl_dsift_delete(dSiftFilt);
  }
  //siftresg[0] has the descriptors now
  for(i=0;i<pre_size;i++) {
    //TODO: optimize the layout
    float halfwidth = ((float)width)/2;
    float halfheight = ((float)height)/2;
    siftframe[0][i*2]= (siftframe[0][i*2]-halfwidth)/width;
    siftframe[0][i*2+1]= (siftframe[0][i*2+1]-halfheight)/height;
  }

  int terms  = pre_size;
  //  cout<<"Terms "<<pre_size<<endl;
  return pre_size;
}

int dsift_gpu(char* imagePath, float** siftresg, float* projection, float* projectionCenter, float** siftframe) {
  //char *imagePath = image;
  double start, finish;
  float duration;
  //  int height, width;
	
  int i, noctaves=4, nlevels=2, o_min=0;
  start = wallclock();
  antialiasing = 1;
  Mat Image = imread(imagePath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  unsigned char Pixel;

  int height = Image.rows;
  int width = Image.cols;
  cout<<"Image "<<imagePath<<", "<<height<<", "<<width<<" Reading "<<wallclock() - start<<endl;

  float scales[NUM_SCALES] = { 1.414213562373095, 1, 0.707106781186548, 0.5, 0.353553390593274, 0.25, 0.176776695296637, 0.125};
  //float *siftres = (float*)malloc(sizeof(float)*height * width*128);
  cv::Mat mtx(height, width, CV_32FC3); 
  int pre_size = 0;
  for(int scale=0;scale<NUM_SCALES;scale++) {
    //    cv::Mat mtx_out(ceil((height)*scales[scale]), ceil(scales[scale]*(width)), CV_32F); 
    int newh = ceil(height*scales[scale]);
    int neww = ceil(width*scales[scale]);
    VlDsiftFilter* dSiftFilt = vl_dsift_new(neww, newh);
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;
    if(framex<=0 || framey<=0 || sizex<=0 || sizey<=0) {
      vl_dsift_delete(dSiftFilt);
      break;
    }

    pre_size += sizey*sizex;
    //WARNING: must delete it!!
    vl_dsift_delete(dSiftFilt);
  }
  int total_size = pre_size;

  float* cur_res;

  for(i=0; i<height; i++)
    {
      for(int j=0; j<width; j++)
	{
	  Pixel = Image.at<unsigned char>(i, j*3); //(unsigned char*)(Image->imageData + (i* Image->width + j)*3);
	  int p1=(int)Pixel;
	  mtx.at<Vec3f>(i,j).val[0]=(float)((int)(Pixel)/255.0);
	  //	  cout<<"Values "<<(int)Pixel<<" ";
	  Pixel = Image.at<unsigned char>(i, j*3+1);
	  int p2 = (int)Pixel;
	  mtx.at<Vec3f>(i,j).val[1]=(float)((int)(Pixel)/255.0);
	  //	  cout<<"Values "<<(int)Pixel<<" ";
	  Pixel = Image.at<unsigned char>(i, j*3+2);
	  mtx.at<Vec3f>(i,j).val[2]=(float)((int)(Pixel)/255.0);
	  /*
      	  if((int)Pixel!=p1 || p1!=p2)
	    cout<<i<<" "<<j<<" Values "<<(int)Pixel<<endl;
	  */
	  //	  cout<<"Image data "<<mtx.at<Vec3f>(i,j).val[0]<<", "<<mtx.at<Vec3f>(i,j).val[1]<<", "<<mtx.at<Vec3f>(i,j).val[2]<<endl;
	}
    }
  //  cout<<"Read OK "<<wallclock() - start<<endl;
  //WARNING: resize if height is bigger than 480
  start  = wallclock();
    
  if(height>480) {
    double scale4 = (double)(480.0/(double)height);
    cv::Mat mtx1(ceil(height*scale4), ceil(scale4*width), CV_32FC3);
    new_resize_C3(mtx, mtx1, scale4);

    mtx = mtx1;
    height = mtx1.rows;
    width = mtx1.cols;
  }
  //  cout<<"Resize Time "<<wallclock() - start<<endl;
  //rgb2gray
  cv::Mat mtx_in(height, width, CV_32F);
  //  float coef[3] = {0.2989, 0.5870, 0.1140};
  //  float coef[3] = {0.1140, 0.5870, 0.2989};
  //  ifstream ff("gpu");
  float coef[3] = {0.114020904255103, 0.587043074451121, 0.298936021293776};
  for(i=0;i<height;i++) {
    for(int j=0;j<width;j++) {
      //      mtx_in.at<float>(i,j) = 0.0;
      //cout<<"In "<<mtx.at<Vec3f>(i,j).val[0]<<endl;
      if(mtx.at<Vec3f>(i,j).val[0] == mtx.at<Vec3f>(i,j).val[1] && mtx.at<Vec3f>(i,j).val[0] == mtx.at<Vec3f>(i,j).val[2])
	mtx_in.at<float>(i,j) = mtx.at<Vec3f>(i,j).val[0];
      else {

	float tmp = 0.0;
	for(int k=0;k<3;k++) {
	  //	mtx_in.at<float>(i,j) += mtx.at<Vec3f>(i,j).val[k]*coef[k];
	  tmp += mtx.at<Vec3f>(i,j).val[k]*coef[k]; //(double)mtx.at<Vec3f>(i,j).val[k] * coef[k];
	}
	mtx_in.at<float>(i,j) = tmp;
      }
    }
  }

  for(int scale=0;scale<NUM_SCALES;scale++) {
    int newh = ceil(height*scales[scale]);
    int neww = ceil(width*scales[scale]);
    VlDsiftFilter* dSiftFilt = vl_dsift_new(neww, newh);
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;
    if(framex<=0 || framey<=0 || sizex<=0 || sizey<=0) {
      vl_dsift_delete(dSiftFilt);
      break;
    }

    pre_size += sizey*sizex;
    //WARNING: must delete it!!
    vl_dsift_delete(dSiftFilt);
  }
  //  int total_size = pre_size;
  //siftresg[0] = (float*) malloc(sizeof(float)*total_size*128);
  //siftframe[0] = (float*) malloc(total_size* sizeof(float) * 2);
  pre_size = 0;
  float* curdata =(float*)mtx_in.data; //(float*) malloc(height*width*sizeof(float));
  for(int scale=0;scale<NUM_SCALES;scale++) {

    start = wallclock();
    int newh = ceil(height * scales[scale]);
    int neww = ceil(width * scales[scale]);
    //cv::Mat mtx_out(ceil((height)*scales[scale]), ceil(scales[scale]*(width)), CV_32F); 
    //    cur_res = (float*)malloc(newh*neww*sizeof(float));
    
    //    new_resize_C1(mtx_in, mtx_out, scales[scale]);

    //    cout<<"Resize time "<<wallclock() - start<<endl;
    VlDsiftFilter* dSiftFilt = vl_dsift_new(neww, newh);
    //1 is inter-linear, 2 is inter-cubic
    //    cur_res = (float*) mtx_out.data;
    sift_time = 0;
    kernel_time = 0;
    copy_time = 0;
    filt_time = 0;
    wait_time = 0;
    init_time = 0;
    int cur_size = pre_size * 128;

    //    sift_time += wallclock() - start;
    //    cout<<"Sift time "<<wallclock() - start1<<endl;
    int frameSizeX = dSiftFilt->geom.binSizeX * (dSiftFilt->geom.numBinX - 1) + 1 ;
    int frameSizeY = dSiftFilt->geom.binSizeY * (dSiftFilt->geom.numBinY - 1) + 1 ;
    int framex =(dSiftFilt->boundMaxX-frameSizeX+2);
    int framey = (dSiftFilt->boundMaxY-frameSizeY+2);
    
    int sizex = framex%dSiftFilt->stepX? framex/dSiftFilt->stepX+1: framex/dSiftFilt->stepX;
    int sizey = framey%dSiftFilt->stepY? framey/dSiftFilt->stepX+1: framey/dSiftFilt->stepY;
    //    cout<<"X and YY "<<sizex<<", "<<sizey<<". "<<framex<<", "<<dSiftFilt->stepX<<endl;
    if(framex<=0 || framey<=0 || sizex<=0 || sizey<=0) {
      cuda_clean();
      vl_dsift_delete(dSiftFilt);
      break;
    }

    gpu_resize(curdata, cur_res, height, width, newh, neww, scales[scale], scale==0, (scale==NUM_SCALES-1), antialiasing);
    
    double    start1 = wallclock();
    my_vl_dsift_process_gpu(dSiftFilt, cur_res,scales[scale], pre_size, (scale==0), (scale==NUM_SCALES-1), total_size, siftresg[0], siftframe[0], scale);

    pre_size += sizey*sizex;
    vl_dsift_delete(dSiftFilt);
  }
  float* dest=(float*)malloc(pre_size*(DST_DIM+2)*sizeof(float));
  
  start = wallclock();
  
    float halfwidth = ((float)width)/2;
    float halfheight = ((float)height)/2;
    gpu_pca_encoding(projection, projectionCenter, dest, pre_size, DST_DIM, height, width, halfheight, halfwidth, siftresg[0]);

  //////////////////
  //  sift_time += wallclock() - start;
    //  float* temp = siftresg[0];
  siftresg[0] = dest;
  int terms  = pre_size;
  //  cout<<"Terms "<<pre_size<<endl;
  return terms;
}

