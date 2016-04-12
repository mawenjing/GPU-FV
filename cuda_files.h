#ifndef GPU_FILES_H
#define GPU_FILES_H

//template <class TT>
//void display(vector<TT>& svec);

#define TYPE float
void gpu_init();
void gpu_copy(TYPE const * covariances, TYPE const * priors, TYPE const * means, int numClusters, int dimension) ;
void gpu_free();
void cuda_clean();
bool gpu_gmm_1(TYPE const * covariances, TYPE const * priors, TYPE const * means, TYPE* posteriors, int numClusters, int dimension, int numData, float halfDimLog2Pi, TYPE* enc_g, TYPE* sqrtInvSigma, TYPE* data);
void saveArr(float* arr, int len, char* filename);
void gpu_sift(float* dst1, float ** src1, int src_height, int src_width, float* const* filtx, float * const* filty, int Wx, int Wy, int binNum, int binsizex, int binsizey, const float* image, float* resg);
double wallclock(void);
void gpu_pca_mm(float* projection, float* projectionCenter, float* data, float* dst, int numData, int dimension);
void gpu_imconvcoltri(float* dst, const float* src, int src_height, int src_width, int binsize, float* w_g, int resized_height, int resized_width);
void gpu_imconvcoltri_fv(float* dst, const float* src, int src_height, int src_width, int binsize, float* w_g, int resized_height, int resized_width, float deltaCenterX, float deltaCenterY, float scale, int offset, int firstflag, int total, int for_training, float* siftframe, int scaleindex);
void gpu_pca_encoding(float* projection, float* projectionCenter, float* dst, int numData, int dimension, int height, int width, float halfheight, float halfwidth, float* input);
void gpu_resize(float* input,float* output, int height, int width, int new_h, int new_w, float scale, int first_scale, int last_scale, int antialiasing);
#endif
