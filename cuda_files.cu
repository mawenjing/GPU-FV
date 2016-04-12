//#include "cutil_inline.h"
//#include "cutil_inline_runtime.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iterator>
#include <cassert>
#include "cuda_files.h"
#include "kernels.cu"
#include "vl/mathop.h"
#include "cublas_v2.h"

#include <cstring>
using namespace std;

#define CUDA_SUCCESS 0
#define THREADS 256
#define THREADS_G 256
#define THREADS2 96
#define THREADS3 96
#define THREADS_S 128
#define ITEMS 4
#define ITEMS3 192 
//#define BLOCKS 16
#define FAILURE 1
#define SUCCESS 0
#define TYPE float
//#define FLT VL_TYPE_FLOAT

//#define VL_GMM_MIN_VARIANCE 2e-6
//#define VL_GMM_MIN_POSTERIOR 1e-2
//#define VL_GMM_MIN_PRIOR 1e-6
#define BINS 8

#define VL_INFINITY_D (vl_infinity_d.value)
TYPE infinity = -(TYPE)VL_INFINITY_D;

#define NOSIFT 0

void gpu_init(){
  int num;
  float* bufx_d;
  cudaGetDeviceCount(&num);
  cudaMalloc((void **)&bufx_d, 1);
  cudaFree(bufx_d);
}
double sift_time, kernel_time, copy_time, filt_time, wait_time, init_time;

TYPE* filtx_d;
TYPE* filty_d;

float* projection_d;
float* projectionCenter_d;
float* tempsum_d;
float* data_d;
float* frames_d;

float* inter1_d;
float* output_d;
float* image_1;
float* w_d;

void gpu_copy(TYPE const * covariances, TYPE const * priors, TYPE const * means, int numClusters, int dimension) {
  int cluster_size = numClusters*sizeof(TYPE);
  int total_size = cluster_size * dimension;

  //TODO: constant memory
  cudaMalloc((void**)&priors_d, numClusters*sizeof(TYPE));
  cudaMalloc((void**)&covariances_d, dimension*numClusters*sizeof(TYPE));
  cudaMalloc((void**)&logWeights_d, numClusters*sizeof(TYPE));
  cudaMalloc((void**)&logCovariances_d, dimension*numClusters*sizeof(TYPE));
  cudaMalloc((void**)&invCovariances_d, dimension*numClusters*sizeof(TYPE));
  cudaMalloc((void**)&means_d, dimension*numClusters*sizeof(TYPE));
  cudaMalloc((void**)&sqrtInvSigma_d, dimension*numClusters*sizeof(TYPE));  

  cudaMemcpy(priors_d, priors, numClusters*sizeof(TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(covariances_d, covariances, total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(means_d, means, total_size, cudaMemcpyHostToDevice);
}

void gpu_free() {
  cudaFree(priors_d);
  cudaFree(covariances_d);
  cudaFree(logWeights_d);
  cudaFree(logCovariances_d);
  cudaFree(invCovariances_d);
  cudaFree(means_d);
  cudaFree(sqrtInvSigma_d);  
}

bool gpu_gmm_1(TYPE const * covariances, TYPE const * priors, TYPE const * means, TYPE* posteriors, int numClusters, int dimension, int numData, float halfDimLog2Pi, TYPE* enc_g, TYPE* sqrtInvSigma, TYPE* data) {
  double start = wallclock();
  int cluster_size = numClusters*sizeof(TYPE);
  int total_size = cluster_size * dimension;
  //  int data_size = numData*numClusters*sizeof(TYPE);
  cudaMalloc((void**)&posteriors_d, numData*numClusters*sizeof(TYPE));
#if NOSIFT
  cudaMalloc((void**)&tmp2_d, sizeof(TYPE)*dimension*numData);
#endif

  /********* Set grid and block ***********/
  int threads1 = THREADS;
  if(numClusters < THREADS) {
    threads1 = numClusters;
  }
  int numblocks1 = numClusters/threads1;
  if(numClusters%threads1) {
    numblocks1 ++;
  }
  //  size_t localThreads[3]  = {threads1, 1, 1};

  dim3 blocks (numblocks1,1, 1);
  dim3 localThreads (threads1, 1, 1);

  int numblocks2 = numData/ITEMS + 1;
  if(numData%ITEMS) {
    numblocks2 ++;
  }
  if(numblocks2 > 65530)
    numblocks2 = 65530;
  dim3 blocks2(numblocks2, 1, 1);
  dim3 localThreads2(THREADS2, 1, 1);

  int numblocks3 = numData/ITEMS3 + 1;
  if(numData%ITEMS3) {
    numblocks3 ++;
  }
  //  cout<<"Total blocks for Kernel 2 "<<numblocks2<<endl;

  dim3 blocks3 (numblocks3, 1, 1);

  //{ globalSizeX, globalSizeY, 1};

  dim3 localThreads3 (THREADS3,1,1);

  cudaMalloc((void**)&enc_d, 2*dimension*numClusters*numblocks3*sizeof(TYPE));
  cudaError_t err = cudaGetLastError();
  if(err!=cudaSuccess)
    cout<<"Malloc Err "<<cudaGetErrorString(err)<<", "<<cudaSuccess<<endl;

  /*********** Set other parameters *********/
#if NOSIFT
  //  for(int i=0;i<100;i++)
  //    cout<<"Then "<<data[i]<<endl;


  cudaMemcpy(tmp2_d, data, numData*dimension*sizeof(TYPE), cudaMemcpyHostToDevice);
  
  err = cudaGetLastError();
  if(err!=cudaSuccess)
    cout<<"Memcpy Err "<<cudaGetErrorString(err)<<", "<<cudaSuccess<<endl;

#endif
  
  /*  cudaMemcpy(data, tmp2_d, numData*dimension*sizeof(TYPE), cudaMemcpyDeviceToHost);
  for(int i=numData*dimension-100;i<numData*dimension;i++)
    cout<<"Then "<<data[i]<<endl;
  */
  /*
  cudaMemcpy(data, priors_d, 256*sizeof(TYPE), cudaMemcpyDeviceToHost);
  for(int i=0;i<256;i++)
    cout<<"Then "<<data[i]<<endl;
  */
  /*
  for(int i=0;i<50;i++) {
    cout<<"data "<<data[i]<<", "<<means[i]<<", "<<priors[i]<<" "<<covariances[i]<<endl;
    }*/
  
  //  TYPE* temp = (TYPE*) malloc(numData*numClusters*sizeof(TYPE));
  
  gmm_1<<<numblocks1, localThreads>>>(covariances_d, invCovariances_d, logCovariances_d, logWeights_d, priors_d, dimension, numClusters, infinity, sqrtInvSigma_d);
  err = cudaGetLastError();
  if(err!=cudaSuccess)
    cout<<"gmm1 Err "<<cudaGetErrorString(err)<<", "<<cudaSuccess<<endl;

  start = wallclock();
  /////////------------------------kernel gmm2
  gmm_2<<<blocks2, localThreads2>>>(invCovariances_d, logCovariances_d, logWeights_d, posteriors_d, numClusters, halfDimLog2Pi, tmp2_d, means_d, infinity, numData);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if(err!=cudaSuccess)
    cout<<"gmm2 Err "<<cudaGetErrorString(err)<<", "<<cudaSuccess<<endl;
  cout<<"Kernel 2 "<<wallclock() - start<<endl;
  /*
  float* temp = (float*)malloc(numData*numClusters*sizeof(float));    
  
  cudaMemcpy(temp, posteriors_d, numData*numClusters * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i=0;i<20*numClusters;i++) 
   cout<<i%numClusters<<" GPU posteriors "<<temp[i]<<endl;
  //   free(temp);
  */
  start = wallclock();
  gmm_3<<<blocks3, localThreads3>>>(enc_d, tmp2_d, means_d, sqrtInvSigma_d, posteriors_d, dimension, numClusters, numData, priors_d);
  /*
  cudaMemcpy(data, priors_d, 100*sizeof(TYPE), cudaMemcpyDeviceToHost);
  for(int i=0;i<100;i++) {
    cout<<"Data "<<data[i]<<endl;
  }
  */
  cudaThreadSynchronize();
  /*
  cudaMemcpy(temp, sqrtInvSigma_d, numClusters*82*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i=0;i<numClusters*82;i++) {
    
    cout<<"Enc "<<temp[i]<<endl;
    }
  */
  err = cudaGetLastError();
  if(err!=cudaSuccess)
    cout<<"gmm3 Err "<<cudaGetErrorString(err)<<", "<<cudaSuccess<<endl;
  cout<<"Kernel 3 "<<wallclock() - start<<endl;
  ////////////
  

  start = wallclock();
  gmm_4<<<numClusters, 128>>>(enc_d, dimension, numClusters, numblocks3);
  err = cudaGetLastError();
  if(err!=cudaSuccess)
    cout<<"gmm4 Err "<<cudaGetErrorString(err)<<", "<<cudaSuccess<<endl;

  cudaThreadSynchronize();
      
  
  //  cout<<"Kernel 4 "<<wallclock() - start<<endl;

#define THREADS_P 128
  int numblocks_p = 28; //numClusters;
  dim3 blocksp(numblocks_p, 1, 1);
  dim3 threadsp(THREADS_P, 1, 1);
  cudaMalloc((void**) &sum_d, numblocks_p*sizeof(TYPE));

  gmm_prefix<<<blocksp, threadsp>>>(enc_d, priors_d, numData, numClusters, dimension, numblocks_p, sum_d);
  float* sum = (float*) malloc(numblocks_p  * sizeof(float));
  cudaMemcpy(sum, sum_d, numblocks_p*sizeof(TYPE), cudaMemcpyDeviceToHost);
    
  float n = 0.0;
  for(int i=0;i<numblocks_p;i++) {
    //    cout<<"Sum is "<<sum[i]<<endl;
    n += sum[i];
  }
  //WARNING: free sum here.
  cout<<"Sum is "<<n<<endl;;

  n = vl_sqrt_f(n) ;
  //n = VL_XCAT(vl_sqrt_, SFX)(n) ;
  n = VL_MAX(n, 1e-12) ;
  free(sum);

  dim3 blocksnorm(numClusters,1 , 1);
  dim3 threadsnorm( 128, 1, 1);
  prefix_norm<<<blocksnorm, threadsnorm>>>(enc_d, numData, numClusters, dimension, numClusters, n);
  cudaMemcpy(enc_g, enc_d, total_size*2, cudaMemcpyDeviceToHost);
  //WARNING: tmp2_d is allocated at sift time
  cudaFree(sum_d);
  cudaFree(enc_d);
  cudaFree(posteriors_d);
  cudaFree(tmp2_d);
  return 0;
}

/////////////////////
/*
__constant__ TYPE* filtx_d;
__constant__ TYPE* filty_d;
*/
void gpu_pca_mm(float* projection, float* projectionCenter, float* data, float* dst, int numData, int dimension) {
  
  cublasHandle_t handle;
  //  checkError(cublasCreate(&handle), "cublasCreate() error!\n");
  cublasCreate(&handle);
  //cout<<"Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int uiHA = 82;
  int uiWA = 128;
  int uiWB= numData;                                                           

  cudaMalloc((void**)&tmp2_d, numData*(2+dimension)*sizeof(float));
  cudaMalloc((void**)&tempsum_d, numData*uiWA*sizeof(float));
  cudaMalloc((void**)&projection_d, (2+dimension)*uiWA*sizeof(float));
  cudaMalloc((void**)&projectionCenter_d, uiWA*sizeof(float));
  cudaMalloc((void**)&data_d, numData*128*sizeof(float));

  //projection2 is 80 * 128, so it is transposed to the following
  float* projection2 = (float*) malloc(82*128*sizeof(float));
  for(int j=0;j<128;j++) {
    for(int i=0;i<80;i++) {
      projection2[j*82+i] = projection[i+j*80];
    }
  }
  for(int j=0;j<128;j++) {
    for(int i=80;i<82;i++) {
      projection2[i+82*j] = 0.0;
    }
  }
  
  cudaMemcpy(projection_d, projection2, 128*82*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(data_d, data, numData*128*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(projectionCenter_d, projectionCenter, 128*sizeof(float), cudaMemcpyHostToDevice);
  dim3 threads(128,1,1);
  dim3 blocks(1024,1,1);
  kernel1<<<blocks, threads>>>(tempsum_d, data_d, projectionCenter_d, numData);
  cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiHA, uiWB, uiWA,&alpha, projection_d, uiHA, tempsum_d, uiWA, &beta, tmp2_d, uiHA);

  //checkError(ret, "cublas Sgemm returned an error!\n");
  cudaMemcpy(dst, tmp2_d, 82*numData*sizeof(float), cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
  free(projection2);
  
  cudaFree(tmp2_d);
  cudaFree(tempsum_d);
  cudaFree(projection_d);
  cudaFree(projectionCenter_d);
  cudaFree(data_d);
}

void gpu_imconvcoltri(float* dst, const float* src, int src_height, int src_width, int binsize, float* w_g, int resized_height, int resized_width) {
  //For the first convoltri,
  //src_height = dstStride == imageHeight, src_width = imageWidth = imageStride;
  int filterSize = binsize;

  cudaMalloc((void**)&grads_d, sizeof(TYPE) * src_height * src_width);

  cudaMalloc((void**)&src_d, sizeof(TYPE) * src_height * src_width * 8);
  cudaMalloc((void**)&tmp1_d, sizeof(TYPE) * src_height * src_width* 8);
  cudaMalloc((void**)&tmp2_d, sizeof(TYPE) * src_height * src_width * 8);

  cudaMalloc((void**)&w_d, sizeof(TYPE) * 16);
  //  cout<<"Error "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
  //TODO: finish this part
  //Initialize the grads
  int error = cudaMemcpy(grads_d, src, src_height*src_width*sizeof(float), cudaMemcpyHostToDevice);
  //cout<<"Error "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
  size_t total_size = 8*src_height*src_width*sizeof(TYPE);
  cudaMemset(src_d, 0, total_size);
  //  cout<<"Error "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
  dim3 threads(THREADS_G, 1, 1);
    //    dim3 blocks(src_height, 1, 1);
  dim3 blocks(src_height, 1, 1);
  init_grads<<<blocks, threads>>>(grads_d, src_d, src_width, src_height);
    
  ///////////////////////  
  double start = wallclock();
  for(int bint=0;bint<1;bint++) {
    cudaMemcpy(dst, src_d, src_height * src_width * sizeof(TYPE), cudaMemcpyDeviceToHost);
    dim3 threads(256, 1, 1);
    dim3 blocks(src_width, 1, 1);

    imconvcoltri<<<blocks, threads>>>(tmp1_d, src_height, src_d+src_width*src_height*bint, src_width, filterSize);
   
    cudaMemcpy(dst+src_height*src_width*0, tmp1_d, src_height * src_width * 8*sizeof(TYPE), cudaMemcpyDeviceToHost);  

    dim3 threads2(256, 1, 1); //src_width + filterSize, 1, 1);
    int block_2 = src_height/4;
    if(src_height%4)
      block_2 ++;

    dim3 blocks2(block_2, 1, 1);
    imconvcoltri2<<<blocks2, threads2>>>(tmp2_d+src_width*src_height*bint, src_width, tmp1_d+src_width*src_height*bint, src_height, filterSize, w_d);
   
    cout<<"Error "<<error<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
    start = wallclock();  
  }

  //Get the final output (4*4 *8)

  float* output_d;
  cudaMalloc(((void**)&output_d), 128* resized_height * resized_width*sizeof(float));
  //cout<<"Err "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;

  cudaMemcpy(w_d, w_g, 16 * sizeof(TYPE), cudaMemcpyHostToDevice);
  /*  
  cout<<"Resized "<<resized_width<<", "<<resized_height<<endl;
  for(int i=0;i<16;i++)
    cout<<"GPU wg is "<<w_g[i]<<endl;
  */
  dim3 threads3(128, 1, 1);
  dim3 blocks3(resized_height, resized_width,1);
  multWX<<<blocks3, threads3>>>(tmp2_d, output_d, w_d, 4, resized_width, resized_height, src_height, src_width);
  //  cout<<"Err "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
  ////////////////////////////////
  ///////////normalization
  //////////////
    int terms = resized_width * resized_height;
    //    cout<<"Total terms "<<terms<<" "<<endl;

    /*    
    cudaMemcpy(dst, output_d, 128*resized_height * resized_width*sizeof(TYPE), cudaMemcpyDeviceToHost);
  for(int i=0;i<1024;i++)
    cout<<"End norm "<<i<<", "<<dst[i]<<endl;
    */

    dim3 localThreadssiftnorm(THREADS_S, 1,1);
    dim3 blockssiftnorm(terms, 1, 1);
    sift_normalize<<<blockssiftnorm, localThreadssiftnorm>>>(output_d, terms);
    //  cout<<"Err "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
  cudaMemcpy(dst, output_d, 128*resized_height * resized_width*sizeof(TYPE), cudaMemcpyDeviceToHost);

  /*    
  for(int i=0;i<128;i++)
    cout<<"End norm "<<i<<", "<<dst[i]<<endl;
  */
    ////////////////////
  //cout<<"Copy time "<<wallclock() - start<<endl;
  cudaFree(grads_d);
  cudaFree(w_d);
  cudaFree(src_d);
  cudaFree(tmp1_d);
  cudaFree(tmp2_d);
  cudaFree(output_d);
}
void gpu_imconvcoltri_fv(float* dst, const float* src, int src_height, int src_width, int binsize, float* w_g, int resized_height, int resized_width, float deltaCenterX, float deltaCenterY, float scale, int offset, int firstflag, int lastflag, int total, float* siftframe, int scaleindex) {
  //For the first convoltri,
  //src_height = dstStride == imageHeight, src_width = imageWidth = imageStride;
  int filterSize = binsize;
  cudaError_t err;

  if(firstflag == 1) {
    cudaMalloc((void**)&src_d, sizeof(TYPE) * src_height * src_width * 8);
    cudaMalloc((void**)&tmp1_d, sizeof(TYPE) * src_height * src_width* 8);
    cudaMalloc((void**)&tmp2_d, sizeof(TYPE) * src_height * src_width * 8);
    cudaMalloc((void**)&w_d, sizeof(TYPE) * 16);
    //cout<<"Allocated\n";
  }
  //cout<<"Error malloc 3 "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;

  size_t total_size = 8*src_height*src_width*sizeof(TYPE);
  cudaMemset(src_d, 0, total_size);
#if 0
  //Use CPU resize
  cudaMalloc((void**)&grads_d, src_width*src_height*sizeof(float));
  cudaMemcpy(grads_d, src, src_width*src_height*sizeof(TYPE), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != CUDA_SUCCESS) {
    cout<<"Error copy "<<cudaGetErrorString(err)<<endl;
  }
  
#endif
  dim3 threads(THREADS_G, 1, 1);
  dim3 blocks(src_height, 1, 1);
  init_grads<<<blocks, threads>>>(grads_d, src_d, src_width, src_height);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if(err != CUDA_SUCCESS) {
    cout<<"Error init "<<cudaGetErrorString(err)<<endl;
  }
  if(firstflag == 1) {
      cudaMalloc(((void**)&frames_d), 2* total*sizeof(float));
      cudaMalloc(((void**)&data_d), 128* total*sizeof(float));
      //cout<<"Err "<<cudaGetErrorString((cudaError_t)cudaGetLastError())<<endl;
      //      cout<<"total "<<total*128*sizeof(float)<<" Got data! "<<data_d<<", "<<frames_d<<", "<<tmp2_d<<"\n";
  }

  for(int bint=0;bint<1;bint++) {
    //    cudaMemcpy(src_d, src[bint], src_height * src_width * sizeof(TYPE), cudaMemcpyHostToDevice);
    //  cudaMemcpy(w_d, w_g, 16 * sizeof(TYPE), cudaMemcpyHostToDevice);
    //    cout<<"Copy 1 time "<<wallclock() - start<<endl;
    //  for(int i=0;i<20;i++)
    //    cout<<"Imput "<<src[i]<<endl;
    
    //dim3 threads(src_height+filterSize, 1, 1);
    dim3 threads(256, 1, 1);
    dim3 blocks(src_width, 1, 1);
    //cout<<"Height "<<src_height<<", "<<bint<<", "<<src_d+src_width*src_height*bint<<endl;
    imconvcoltri<<<blocks, threads>>>(tmp1_d, src_height, src_d+src_width*src_height*bint, src_width, filterSize);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if(err != CUDA_SUCCESS) {
      cout<<"Error imconv 1 "<<cudaGetErrorString(err)<<endl;
    }

    //cudaThreadSynchronize();

    dim3 threads2(256, 1, 1); //src_width + filterSize, 1, 1);
    int block_2 = src_height/4;
    if(src_height%4)
      block_2 ++;

    ///////////
    //block_2 = src_height;
    dim3 blocks2(block_2, 1, 1);
    imconvcoltri2<<<blocks2, threads2>>>(tmp2_d+src_width*src_height*bint, src_width, tmp1_d+src_width*src_height*bint, src_height, filterSize, w_d);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if(err != CUDA_SUCCESS) {
      cout<<"Error imconv 2 "<<cudaGetErrorString(err)<<endl;
    }
  }

  //Get the final output (4*4 *8)
      
  //WARNING: allocate frames_d here, need to free it later!
  
  cudaMemcpy(w_d, w_g, 16 * sizeof(TYPE), cudaMemcpyHostToDevice);
  
  /*  
  cout<<"Resized "<<resized_width<<", "<<resized_height<<endl;
  for(int i=0;i<16;i++)
    cout<<"GPU wg is "<<w_g[i]<<endl;
  */
  dim3 threads3(128, 1, 1);
  dim3 blocks3(resized_width, resized_height,1);
  //  cout<<"Blocks 3 "<<resized_width<<", "<<resized_height<<endl;
  multWX<<<blocks3, threads3>>>(tmp2_d, data_d+offset*128, w_d, 4, resized_height, resized_width, src_height, src_width);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if(err != CUDA_SUCCESS) {
    cout<<"Error multWX 2 "<<cudaGetErrorString(err)<<endl;
  }

  ////////////////////////////////
  ///////////normalization
  //////////////
  int terms = resized_width * resized_height;
  //cout<<"Total terms "<<terms<<" "<<endl;

    /*    
    cudaMemcpy(dst, output_d, 128*resized_height * resized_width*sizeof(TYPE), cudaMemcpyDeviceToHost);
  for(int i=0;i<1024;i++)
    cout<<"End norm "<<i<<", "<<dst[i]<<endl;
    */
  double start = wallclock();
  dim3 localThreadssiftnorm(THREADS_S, 1,1);
  //WARNING: terms could be larger than 65536
  dim3 blockssiftnorm(32768, 1, 1);

  sift_normalize<<<blockssiftnorm, localThreadssiftnorm>>>(data_d+offset*128, terms);
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if(err != CUDA_SUCCESS) {
    cout<<"Error sift norm "<<cudaGetErrorString(err)<<endl;
  }
  //cout<<"Sift norm "<<wallclock() - start<<endl;
  /*
  float* data = (float*)malloc(terms*128*sizeof(float));
  cudaMemcpy(data, data_d+offset*128, terms*128*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i=0;i<terms;i++) {
    cout<<i<<" Before pca "<<data[i*128]<<endl;
  }
  free(data);
  */

  //  cudaMemcpy(dst+offset*128, data_d+offset*128, terms*128*sizeof(float), cudaMemcpyDeviceToHost);

  int total_frames = 2*resized_height*resized_width;
  int THREADSF = 256;
  dim3 threadsf(THREADSF, 1, 1);
  int blocks_f = total_frames/THREADSF;
  if(total_frames%THREADSF!=0) {
    blocks_f ++;
  }
  
  dim3 blocksf(blocks_f, 1, 1);
      //TODO: pass this parameter
  int step = 4;
//  cout<<"Center "<<deltaCenterX<<", "<<deltaCenterY<<endl;
  get_frame<<<blocksf, threadsf>>>(deltaCenterX, deltaCenterY, frames_d+2*offset, total_frames, resized_width, resized_height, step, scale);
  cudaThreadSynchronize();
    err = cudaGetLastError();
    if(err != CUDA_SUCCESS) {
      cout<<"Error get frame "<<cudaGetErrorString(err)<<endl;
    }
    //  cudaMemcpy(siftframe+offset*2, frames_d+offset*2, terms*2*sizeof(float), cudaMemcpyDeviceToHost);

  if(lastflag == 1) {
    cudaFree(grads_d);
    cudaFree(w_d);
    cudaFree(src_d);
    cudaFree(tmp1_d);
    cudaFree(tmp2_d);
  }
  //  cudaFree(output_d);
}
void gpu_pca_encoding(float* projection, float* projectionCenter, float* dst, int numData, int dimension, int height, int width, float halfheight, float halfwidth, float* input) {
  
  cublasHandle_t handle;
  //  checkError(cublasCreate(&handle), "cublasCreate() error!\n");
  cublasCreate(&handle);
  cout<<"Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int uiHA = 82;
  int uiWA = 128;
  int uiWB= numData;                                                           

  cudaMalloc((void**)&tmp2_d, numData*(2+dimension)*sizeof(float));
  //cout<<"Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
  cudaMalloc((void**)&tempsum_d, numData*uiWA*sizeof(float));
  cudaMalloc((void**)&projection_d, (2+dimension)*uiWA*sizeof(float));
  cudaMalloc((void**)&projectionCenter_d, uiWA*sizeof(float));
  //  cudaMalloc((void**)&data_d, numData*128*sizeof(float));

  //projection2 is 80 * 128, so it is transposed to the following
  float* projection2 = (float*) malloc(82*128*sizeof(float));
  for(int j=0;j<128;j++) {
    for(int i=0;i<80;i++) {
      //      projection2[j*82+i] = projection[i*128+j];
      projection2[j*82+i] = projection[i+j*80];
    }
  }
  for(int j=0;j<128;j++) {
    for(int i=80;i<82;i++) {
      projection2[i+82*j] = 0.0;
    }
  }
  
  cudaMemcpy(projection_d, projection2, 128*82*sizeof(float), cudaMemcpyHostToDevice);
  //WARNING: the data is kept in GPU memroy
  //  cudaMemcpy(data_d, input, numData*128*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(projectionCenter_d, projectionCenter, 128*sizeof(float), cudaMemcpyHostToDevice);
  dim3 threads(128,1,1);
  dim3 blocks(1024,1,1);
  kernel1<<<blocks, threads>>>(tempsum_d, data_d, projectionCenter_d, numData);
  //    if(cudaGetLastError()!=cudaSuccess)
  //  cout<<"kernel 1 Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;

    //should use dest = tempsum(numData*128) * projection(128*80);
    //should use dest = projection(128*80) * tempsum(numData*128);
  //  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWA);

    //    cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiHA, uiWB, uiWA,&alpha, tempsum_d, uiHA, projection_d, uiWA, &beta, tmp, uiHA);
  cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiHA, uiWB, uiWA,&alpha, projection_d, uiHA, tempsum_d, uiWA, &beta, tmp2_d, uiHA);
  //  cout<<"cublas Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;

  //Add siftframe
  int total = numData*2;
  int THREADSF = 256;
  dim3 threadsf(THREADSF, 1, 1);
  int blocks_f = total/THREADSF;
  if(blocks_f%THREADSF!=0) {
    blocks_f ++;
  }
  dim3 blocksf(blocks_f, 1, 1);

  add_frame<<<blocksf, threadsf>>>(tmp2_d, frames_d,(float)width, (float)height, halfwidth, halfheight, total);

  //#if NOSIFT
#if 0
  cudaMemcpy(dst, tmp2_d, 82*numData*sizeof(float), cudaMemcpyDeviceToHost);
  ifstream code("feature");
  float* temp = (float*)malloc(82*numData*sizeof(float));
  int i=0;
  
  while(code>>temp[i++]) {
  }
  code.close();

  for(int i=0;i<numData*82;i++) {
    if( abs(temp[i] - dst[i]) > 0.00001) 
      cout<<i<<" Tmp 2 "<<dst[i]<<", should be "<<temp[i]<<endl;
      }
#endif
  //  cudaFree(tmp2_d);
  //#endif

  cublasDestroy(handle);
  free(projection2);
  
  //  cudaFree(tmp2_d);
  cudaFree(tempsum_d);
  cudaFree(projection_d);
  cudaFree(projectionCenter_d);
  cudaFree(data_d);
  cudaFree(frames_d);
}


void gpu_resize(float* input,float* output, int height, int width, int new_h, int new_w, float scale, int first_scale, int last_scale, int antialiasing) {
  dim3 threads(256, 1, 1);
  dim3 blocks(new_h, 1,1);
  if(first_scale) {
    cudaMalloc((void**)&image_1,  height*width*sizeof(float));
    cudaMemcpy(image_1, input, height*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&inter1_d, new_h*new_w*sizeof(float));
    cudaMalloc((void**)&grads_d, new_h*new_w*sizeof(float));
  }
  float kernelwidth;
  if(antialiasing == 1 && scale<1) 
    kernelwidth = 2/scale;
  else kernelwidth = 2;

  resize_h<<<blocks, threads>>>(inter1_d, image_1, height, width, scale, kernelwidth, antialiasing);
  //cudaThreadSynchronize();
  //  cout<<"Err resize "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
  dim3 blocks2(new_w, 1, 1);
  resize_w<<<blocks2, threads>>>(grads_d, inter1_d, height, width, scale, new_h, new_w, kernelwidth,antialiasing);
  //cudaThreadSynchronize();
  
  //cout<<"Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
  //  for(int i=0;i<100;i++)

  //  output = (float*) malloc(new_h*new_w*sizeof(float));

  //  cudaMemcpy(output, grads_d, new_h*new_w*sizeof(float), cudaMemcpyDeviceToHost);
  /*
  cout<<"My resize "<<new_w<<", "<<new_h<<endl;
  for(int i=0;i<new_w*new_h;i++) //new_w*height;i<height*new_w+new_w;i++)
    cout<<i<<" Resized "<<output[i]<<endl;
  */
  //  cudaFree(output_d);
  if(last_scale) {
    cudaFree(image_1);
    cudaFree(inter1_d);
    //cout<<"It is lost "<<endl;
  }
}

void cuda_clean() {
  cudaFree(image_1);
  //  cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
  cudaFree(inter1_d);
  //cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
    cudaFree(grads_d);
    //cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
    cudaFree(w_d);
    //cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
    cudaFree(src_d);
    //cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
    cudaFree(tmp1_d);
    //cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
    cudaFree(tmp2_d);
    //cout<<"clean Err "<<cudaGetErrorString(cudaGetLastError())<<", "<<cudaSuccess<<endl;
}
void gpu_sift(float* dst1,  float ** src1,  int src_height, int src_width, float *const* filtx, float *const* filty, int Wx, int Wy, int binNum, int bin_sizex, int bin_sizey, const float* image, float* resg) {
  //step is 1, flags is non-zero-pad && transpose
  double start, finish, start1;//, start2;
  int i;
  //TODO: specify scales, and step
  float scales[9] = {1, 2, 1, 0.5};
  int sizesx[9], sizesy[9];
  int offsetssrc[9], offsetstmp[9], offsetsdst[9];
  int step = 4;

  int filtx_size, filty_size, src_size=0, dst_size=0;
  cudaMalloc((void**)&image_d, sizeof(TYPE)*src_height * src_width);
  cudaMemcpy(image_d, image, src_height * src_width *sizeof(float), cudaMemcpyHostToDevice);

  for(i=0;i<1;i++) {
    sizesx[i] = ceil(src_width * scales[i]);
    sizesy[i] = ceil(src_height * scales[i]);
    src_size += sizesx[i] * sizesy[i];
    dst_size += (sizesy[i]/step+1)*(sizesx[i]/step+1);
    
    offsetssrc[i] = (src_size - sizesx[i] * sizesy[i])*8;
    offsetstmp[i] = (src_size - sizesx[i] * sizesy[i]) *32;
    offsetsdst[i] = (dst_size -  (sizesy[i]/step+1)*(sizesx[i]/step+1)) *128;
  }

  //  float duration;
  start = wallclock();
  filtx_size = 2*bin_sizex ;
  filty_size = 2*bin_sizey ;

  start1 = wallclock();
  cudaMalloc((void**)&filtx_d, filtx_size*4*sizeof(float));
  cudaMalloc((void**)&filty_d, filty_size*4*sizeof(float));

  for(i=0;i<4;i++) {
    cudaMemcpy(filtx_d+filtx_size*i, filtx[i], (filtx_size-1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filty_d+filty_size*i, filty[i], (filty_size-1)*sizeof(float), cudaMemcpyHostToDevice);
  }  
  
  start1 = wallclock();


  cudaMalloc((void**)&tmp2_d, sizeof(TYPE)*dst_size*128);
  cudaMalloc((void**)&grads_d, sizeof(TYPE)*src_size);
  cudaMalloc((void**)&src_d, sizeof(TYPE)*src_size*8);
  cudaMalloc((void**)&tmp1_d, sizeof(TYPE)*src_size*32);

  //EXPENSIVE
  copy_time += wallclock() - start1; 

  ///////////////////////////////////
  //////////////////////initialize grads///////////////
  /////////////////////////////////////////////
  
  
#define THREADX 128
#define THREADY 128
  for(int j = 0; j < 1; j++) {
    //    float scale = scales[j];
    dim3 threadsi(THREADX, THREADY, 1);
    int blockx = sizesx[j]/THREADX;
    if(sizesx[j]%THREADS) 
      blockx++;
    int blocky = sizesy[j]/THREADY;
    if(sizesy[j]%THREADY)
      blocky++;

    dim3 blocksi(blockx, blocky, 1);
    //int dst_rows = sizesy[j];
    //    int dst_cols = sizesx[j];
    //int src_cols = src_width;
    //int src_rows = src_height;

    //    resize<<<blocksi, threadsi>>>(grads_d+offsetssrc[j], image_d,0, 0, src_width, src_width, src_cols, src_rows, dst_cols, dst_rows, scale, scale);

    int error = cudaMemcpy(grads_d, image_d, src_height*src_width*sizeof(float), cudaMemcpyDeviceToDevice);


    /*    error = cudaMemcpy(resg, image_d, src_height * src_width *sizeof(float), cudaMemcpyDeviceToHost);
    cout<<"Error "<<error<<cudaGetErrorString((cudaError_t)error)<<endl;
  for(i=0;i<200;i++)
    cout<<"Before no "<<image[i]<<", "<<resg[i]<<endl;
    */
    dim3 threads(THREADS_G, 1, 1);
    //    dim3 blocks(src_height, 1, 1);
    dim3 blocks(sizesy[j], 1, 1);
    cout<<"Let's do "<<offsetssrc[j]<<", "<<sizesx[j]<<", "<<sizesy[j]<<endl;    
    init_grads<<<blocks, threads>>>(grads_d+offsetssrc[j], src_d+offsetssrc[j], sizesx[j], sizesy[j]);

    int numblocks1 = src_width; ///THREADS;
    dim3 localThreads(THREADS_S, 1, 1);
    dim3 blockssift (numblocks1, 1, 1);

    sift<<<blockssift, localThreads>>>(tmp1_d+offsetstmp[j], sizesy[j], src_d+offsetssrc[j],sizesx[j], sizesy[j],sizesx[j], filty_d, Wy);
 
    numblocks1 = sizesy[j]/step + 1; //src_height/step + 1;
    
    dim3 localThreadssift2 (THREADS_S, 1, 1);
    dim3 blockssift2(numblocks1, 1, 1);
  
    sift2<<<blockssift2, localThreadssift2>>>(tmp2_d+offsetsdst[j], sizesx[j], tmp1_d+offsetstmp[j], sizesy[j], sizesx[j], sizesy[j], filtx_d, Wx);


    kernel_time += wallclock() - start1; 

    //    start2 = wallclock();
  /////////////////////////////////////////
  //////////////normalize histogram
  ///////////////////////////
    int terms1 = (sizesy[j]-25)/step;
    //    if((sizesy[j] -24)%step)
      terms1 ++;
    int terms2 = (sizesx[j]-25)/step;
    //if((sizesx[j]- 24)%step)
      terms2++;

    int terms = terms1 * terms2;
    cout<<"Total terms "<<terms<<" "<<endl;

    dim3 localThreadssiftnorm(THREADS_S, 1,1);
    dim3 blockssiftnorm(terms, 1, 1);
    sift_normalize<<<blockssiftnorm, localThreadssiftnorm>>>(tmp2_d+offsetsdst[j], terms);
    cout<<"End of normalize"<<endl;
  }
  double start3 = wallclock();
#if 1
  cudaMemcpy(resg, tmp2_d, dst_size*128*sizeof(float), cudaMemcpyDeviceToHost);
#endif
  //EXPENSIVE
  cout<<"Copy time (nouse) "<<wallclock() - start3<<endl;
  cudaFree(image_d);
  cudaFree(src_d);
  cudaFree(grads_d);
  cudaFree(filtx_d);
  cudaFree(filty_d);
  cudaFree(tmp1_d);
  /*
  clReleaseMemObject(image_d);
  clReleaseMemObject(src_d[0]);
  clReleaseMemObject(tmp1_d[0]);
  
  clReleaseMemObject(filtx_d);
  clReleaseMemObject(filty_d);
  */
  
  finish = wallclock();
  sift_time += finish-start;
}
