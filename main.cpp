#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <set>
#include <iterator>
#include <cassert>
#include <dirent.h>
#include "pca.h"
#include <random>
//#include "sift.h"
#include "mydsift.h"
//#include "CommonTools.h"
#include "omp.h"

//#include "CL/cl.h"
//#include "Timer.h"
#include "cuda_files.h"

#include <cstring>
//#include "SDKFile.hpp"
extern "C"{
#include "vl/generic.h"
#include "vl/gmm.h"
#include "vl/fisher.h"
#include "vl/mathop.h"
}

#include "my_svmtrain.h"

using namespace std;

int N = 256;
#define THREADS 256
#define THREADS2 128
#define THREADS3 128

#define ITEMS 4
#define ITEMS3 1024
//#define BLOCKS 16
#define FAILURE 1
#define SUCCESS 0
#define TYPE float
//#define FLT VL_TYPE_FLOAT

#define VL_GMM_MIN_VARIANCE 2e-6
#define VL_GMM_MIN_POSTERIOR 1e-2
#define VL_GMM_MIN_PRIOR 1e-6
#define GPU 1
double gpu_time;
#define SIZE 41984
double file_time;

void saveArr(float* arr, int len, char* filename){
  FILE *fp;
  //WARNING: Use 'a' for file writing
  assert( (fp=fopen(filename, "w")) && "File open eror!!" ); 
  int i;
  for(i=0; i<len; i++)
    {
      //      cout<<"Write "<<i<<", "<<arr[i]<<endl;
      fprintf(fp, "%lf ", arr[i]);
    }
  fprintf(fp, "\n");
  fclose(fp);
}

/////////////////////

double
my_get_gmm_data_posteriors_f(TYPE * posteriors,
			     vl_size numClusters,
			     vl_size numData,
			     TYPE const * priors,
			     TYPE const * means,
			     vl_size dimension,
			     TYPE const * covariances,
			     TYPE const * data, 
			     TYPE * enc,
			     TYPE* sqrtInvSigma)
{
  vl_index i_d, i_cl;
  vl_size dim;
  double LL = 0;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * logCovariances ;
  TYPE * logWeights ;
  TYPE * invCovariances ;
  TYPE * posteriors_g = (TYPE*) vl_malloc(sizeof(TYPE)* numClusters * numData);
  cout<<"Start gmm----------- "<<numClusters<<", numData "<<numData<<endl;
	
  //#if (FLT == VL_TYPE_FLOAT)
  VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
  /*
#else
  VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif
  */
  logCovariances = (TYPE*)vl_malloc(sizeof(TYPE) * numClusters) ;
  invCovariances = (TYPE*)vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
  logWeights = (TYPE*)vl_malloc(sizeof(TYPE) * numClusters) ;

  //  vl_set_num_threads(16);
  //#undef _OPENMP
  //#define _OPENMP 1
#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads() * 1)
#endif
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE logSigma = 0 ;
    if (priors[i_cl] < VL_GMM_MIN_PRIOR) {
      logWeights[i_cl] = - (TYPE) VL_INFINITY_D ;
    } else {
      logWeights[i_cl] = log(priors[i_cl]);
    }
    for(dim = 0 ; dim < dimension ; ++ dim) {
      logSigma += log(covariances[i_cl*dimension + dim]);
      invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
    }
    logCovariances[i_cl] = logSigma;
    //    cout<<"Threads "<<omp_get_num_threads()<<", "<<omp_get_thread_num()<<endl;
  } /* end of parallel region */
  /*    
	for(int i=0;i<300;i++) {
	cout<<"Cpu covariance "<<i<<" "<<invCovariances[i]<<endl;
	}
  */


#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL)	\
  num_threads(vl_get_max_threads()*1)
#endif
  for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
    TYPE clusterPosteriorsSum = 0;
    TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
      TYPE temp = distFn(dimension,
						     data+i_d*dimension,
						     means+i_cl*dimension,
						     invCovariances+i_cl*dimension);
      TYPE p =
	logWeights[i_cl]
	- halfDimLog2Pi
	- 0.5 * logCovariances[i_cl]
	- 0.5 * temp;

      posteriors[i_cl + i_d * numClusters] = p ;
      if (p > maxPosterior) { maxPosterior = p ; }
    }

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      TYPE p = posteriors[i_cl + i_d * numClusters] ;
      p =  exp(p - maxPosterior) ;
      posteriors[i_cl + i_d * numClusters] = p ;
      clusterPosteriorsSum += p ;
    }
    //    LL +=  log(clusterPosteriorsSum) + (double) maxPosterior ;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      //      cout<<"Result "<<clusterPosteriorsSum<<" "<<posteriors[i_cl+i_d*numClusters]<<endl;
      posteriors[i_cl + i_d * numClusters] /= clusterPosteriorsSum ;
    }
  } /* end of parallel region */

  vl_free(logCovariances);
  vl_free(logWeights);
  vl_free(invCovariances);

  return LL;
}

void
my_get_gmm_data_posteriors_f_gpu(TYPE * posteriors,
				 vl_size numClusters,
				 vl_size numData,
				 TYPE const * priors,
				 TYPE const * means,
				 vl_size dimension,
				 TYPE const * covariances,
				 TYPE * enc,
				 TYPE* sqrtInvSigma, TYPE* data)
{
  vl_index i_d, i_cl;
  vl_size dim;
  double LL = 0;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * logCovariances ;
  TYPE * logWeights ;
  TYPE * invCovariances ;
  TYPE * posteriors_g; // = (TYPE*) vl_malloc(sizeof(TYPE)* numClusters * numData);

  double start,finish;
  float duration;
  start = wallclock();
  gpu_gmm_1( covariances,  priors, means, posteriors_g, numClusters, dimension, numData, halfDimLog2Pi, enc, sqrtInvSigma, data) ;
}

int my_fisher_encode(	
		     TYPE * enc,
		     TYPE const * means, vl_size dimension, vl_size numClusters,
		     TYPE const * covariances,
		     TYPE const * priors,
		     TYPE const * data, vl_size numData, int flags) {

  vl_size dim;
  vl_index i_cl, i_d;
  vl_size numTerms = 0 ;
  TYPE * posteriors ;
  TYPE * sqrtInvSigma;

  assert(numClusters >= 1) ;
  assert(dimension >= 1) ;

  posteriors = (TYPE*)vl_malloc(sizeof(TYPE) * numClusters * numData);
  sqrtInvSigma = (TYPE*)vl_malloc(sizeof(TYPE) * dimension * numClusters);
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
    for(dim = 0; dim < dimension; dim++) {
      sqrtInvSigma[i_cl*dimension + dim] = sqrt(1.0 / covariances[i_cl*dimension + dim]);
    }
  }
  clock_t start,finish;
  float duration;
  start = clock();
  //  cout<<"Num data "<<numData<<", "<<dimension<<", "<<numClusters<<endl;

  my_get_gmm_data_posteriors_f(posteriors, numClusters, numData,
			       priors,
			       means, dimension,
			       covariances,
			       data, enc, sqrtInvSigma) ;
  
#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(i_cl, i_d, dim) num_threads(vl_get_max_threads()) reduction(+:numTerms)
#endif

  for(i_cl = 0; i_cl < (signed)numClusters; ++ i_cl) {
    TYPE uprefix;
    TYPE vprefix;

    TYPE * uk = enc + i_cl*dimension ;
    TYPE * vk = enc + i_cl*dimension + numClusters * dimension ;
    //WARNING: new code
    if (priors[i_cl] < 1e-6) { continue ; }

    for(i_d = 0; i_d < (signed)numData; i_d++) {
      TYPE p = posteriors[i_cl + i_d * numClusters] ;

      if (p < 1e-6) continue ;
      numTerms += 1;
      for(dim = 0; dim < dimension; dim++) {
        TYPE diff = data[i_d*dimension + dim] - means[i_cl*dimension + dim] ;
        diff *= sqrtInvSigma[i_cl*dimension + dim] ;
        *(uk + dim) += p * diff ;
        *(vk + dim) += p * (diff * diff - 1);
      }
    }
    uprefix = 1/(numData*sqrt(priors[i_cl]));
    vprefix = 1/(numData*sqrt(2*priors[i_cl]));

    for(dim = 0; dim < dimension; dim++) {
      *(uk + dim) = *(uk + dim) * uprefix;
      *(vk + dim) = *(vk + dim) * vprefix;
    }
  }
  cout<<"Num terms "<<numTerms<<endl;
#if 1
  vl_free(posteriors);
  vl_free(sqrtInvSigma) ;
  finish = clock();
  duration=(float)(finish-start)/CLOCKS_PER_SEC;
  double cpu_time = duration-gpu_time;
  printf("CPU gmm_post time = %.4f, speedup %.2f\n", cpu_time, cpu_time/gpu_time);

  //Wenjing: omit FLAG_SQUARE_ROOT for now
  start = clock();
  //  cout<<"???? "<<flags<<" "<<VL_FISHER_FLAG_SQUARE_ROOT<<endl;
  
  //WARNING: should have sqrt!
  //  if (flags & VL_FISHER_FLAG_SQUARE_ROOT) 
  {
    for(dim = 0; dim < 2 * dimension * numClusters ; dim++) {
      TYPE z = enc [dim] ;
      if (z >= 0) {
        enc[dim] = vl_sqrt_f(z); //VL_XCAT(vl_sqrt_, SFX)(z) ;
      } else {
        enc[dim] = -vl_sqrt_f(z); // VL_XCAT(vl_sqrt_, SFX)(- z) ;
      }
    }
  }

  if (flags & VL_FISHER_FLAG_NORMALIZED) {
    TYPE n = 0 ;
    for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
      TYPE z = enc [dim] ;
      n += z * z ;
    }
    n = vl_sqrt_f(n) ;
    //n = VL_XCAT(vl_sqrt_, SFX)(n) ;
    n = VL_MAX(n, 1e-12) ;
    cout<<"CPU sum is "<<n<<endl;
    for(dim = 0 ; dim < 2 * dimension * numClusters ; dim++) {
      enc[dim] /= n ;
      //      cout<<"Res "<<dim<<", "<<enc[dim]<<endl;
#if 0
      if( abs(enc[dim] - enc_g[dim]) > 0.0001) 
	{
	  cout<<"Wrong "<<dim<<" right "<<enc[dim]<<", my "<<enc_g[dim]<<endl;
	}
#endif
    }
  }
#endif
  
  finish = clock();
  duration=(double)(finish-start)/CLOCKS_PER_SEC;

  cout<<"Normalization time "<<duration<<endl;

  return numTerms ;
};

int my_fisher_encode_gpu (	
			  TYPE * enc_g,
			  TYPE const * means, vl_size dimension, vl_size numClusters,
			  TYPE const * covariances,
			  TYPE const * priors,
			  vl_size numData, TYPE* data) {

  vl_size dim;
  vl_index i_cl, i_d;
  vl_size numTerms = 0 ;
  TYPE * posteriors ;
  TYPE * sqrtInvSigma;

  assert(numClusters >= 1) ;
  assert(dimension >= 1) ;

  my_get_gmm_data_posteriors_f_gpu(posteriors, numClusters, numData,
				   priors,
				   means, dimension,
				   covariances,
				   enc_g, sqrtInvSigma, data) ;
  return numTerms ;
};

int onlineProcessing(char* image, float* priors, float* means, float* covariances, float* enc, float* projection, float* projectionCenter, char* name) { //, float* vobdata, int numBOV){
  double start,finish;
  double durationsift, durationgmm;
  /////////////////////////////////////////////////////////
  ///////////////////////////////gpu version////////////////
  /////////////////////////////////////////////////////
  //step0: load gmm models
  //step1: sift extraction 
  int siftResult;
  cout<<"Process "<<image<<endl;
  start = wallclock();
  float *siftresg;
  float* siftframe;
  // siftResult = dsift_cpu(image, &siftresg, projection, projectionCenter);
  siftResult = dsift_gpu(image, &siftresg, projection, projectionCenter, &siftframe);
  finish = wallclock();

  durationsift=(double)(finish-start);
  cout<<image<<"\nGPU sift time:"<<durationsift<<endl;

  /*  float* siftresc;
  siftResult = dsift_cpu(image, &siftresc, projection, projectionCenter); //, tt, height, width);
  cout<<"Finish 2 "<<siftResult<<endl;
  for(int i=0;i<siftResult;i++) {
    if(abs(siftresg[i]-siftresc[i])>0.001) {
      cout<<"Bad "<<siftresg[i]<<", should be "<<siftresc[i]<<endl;
    }
    }*/
  int dimension = 82;
  int numClusters = N;
  //  TYPE *encg = (TYPE*)vl_malloc(sizeof(float)*2 * dimension * numClusters);

  /////////////////////////////////////////////////////
  /////////////////////////////////CPU version//////////////////////
  ////////////////////////////////////////////

  //  vector<vector<float> > fvec;
  start = wallclock();

  //step2: calculating gmm and fisher and normalization
  
  start = wallclock();
  memset(enc, 0, sizeof(float)*2*dimension*numClusters);

  //  siftresg = (float*)malloc(sizeof(float)*128);
  int res;
#if 1
  //defined(GPU) 
  res = (int)my_fisher_encode_gpu((TYPE*)enc, means, 
				      dimension, numClusters,
				      covariances, priors,
				      siftResult,siftresg);
  //     VL_FISHER_FLAG_NORMALIZED);
  finish = wallclock();
  durationgmm=(double)(finish-start);
  cout<<"GPU "<<image<<" fisher time(s):"<<durationgmm<<endl;

#else 
  res = (int)my_fisher_encode((TYPE*)enc, means, 
				  dimension, numClusters,
				  covariances, priors,
				  siftresg, siftResult,
				  VL_FISHER_FLAG_NORMALIZED);
  cout<<"Fisher time "<<wallclock() - start<<endl;
#endif

  ///////////WARNING: add the other NOOP
  float sum = 0.0;
  for(int i=0;i<SIZE;i++) {
    sum += enc[i] * enc[i];
  }
  for(int i=0;i<SIZE;i++) {
    //WARNING: didn't use the max operation
    enc[i]/=sqrt(sum);
  }
  sum = 0.0;
  for(int i=0;i<SIZE;i++) {
    sum += enc[i] * enc[i];
  }
  for(int i=0;i<SIZE;i++) {
    //WARNING: didn't use the max operation
    enc[i] /=sqrt(sum);
  }

  free(siftresg);
#if SAVE_CODE
  start = wallclock();

  saveArr((float*) enc, 2 * dimension * numClusters, name);
  file_time += wallclock() - start;
  cout<<"Time to save "<<wallclock() - start<<endl;
#endif
  return 0;
}


template <class TT>
void copyVec2Arr(vector<TT> fvec, TT* arr){
  typename vector<TT>::iterator iter = fvec.begin();
  int index = 0;
  for(iter;iter != fvec.end(); iter++)
    {
      arr[index++] = *iter;
    }
}


int VL_CAT(fun, f)(int a, int b)
{
  return a+b;
}

//TRAINSET is the set of train and valication
//TESTSET is the test set 
#define TRAINSET 7680
#define TESTSET 6400

//TRAIN is the number of images to train vl cluster
#define TRAIN_GMM_SET 5000
//number of classes
#define CLASSES 256

#define MAX_PATHH 256

bool mycompare(char* x, char* y) {
  //  cout<<"Compare "<<x<<", "<<y<<endl;
  if(strcmp(x, y)<=0) return 1;
  else return 0;
}

int main(int argc, char* argv[])
{
  //assert(argc==1);
  file_time = 0;
  clock_t start,finish;
  double duration;
  int numData, numBOV;

  float *means, *covariances, *priors, *posteriors;
  vl_size dimension = 82, numComponents=N, numClusters=N;
  //	vl_size dimension = 128, numComponents=512, numClusters=512;

#if defined(GPU)
  gpu_init();
#endif


  if(argc<2) {
    printf("Usage: ./main 0 <file>, where <file> is the dictionary for training the clusters.\n");
    printf("Usage: ./main 1 <input> <output> is for generating fisher vectors.\n");
    return 0;
  }

  int i=0;
  cout<<endl<<"----------------"<<endl;
  string temptrain;
  char nametrain[128];
  

  //  ifstream train_file("/media/sdb1/yulei/VOCdevkit/VOC2007_train/ImageSets/Main/trainval.txt");
  //  ifstream val_file("/media/sdb1/yulei/VOCdevkit/VOC2007_train/ImageSets/Main/val.txt");

  float* sift_res;
  float* sift_frame;

  //WARNING: need to calculate this number  
  int numImages = ceil((float)numComponents * 1000.0/(float)TRAIN_GMM_SET);

  float* final_res = (float*) malloc(numImages*TRAIN_GMM_SET*128*sizeof(float));
  float* final_frame = (float*) malloc(numImages*TRAIN_GMM_SET*128*sizeof(float));

  //  int fileindex[TRAIN] = {1, 3, 4, 6, 7, 9, 10,12, 13, 15};

  //The list of image file names
  //TODO: separate train and test list (train and validation sets comprise the training set
  vector<char*> whole_list[CLASSES];
  int class_files[CLASSES];
  const char* home=argv[2]; //"/home/wenjing/vlfeat/apps/recognition/data/caltech256/256_ObjectCategories";
  DIR* d;
  struct dirent* cur_dir;
  d = opendir(home);
  i=0;
  vector<char*> paths;  
  //  vector<char*> sorted_paths;
  while ((cur_dir = readdir(d)) != NULL)
    {
      if ((strcmp( cur_dir->d_name, "." ) != 0) && (strcmp( cur_dir->d_name, ".." ) != 0))
	{
	  //  char szTempDir[MAX_PATHH] = { 0 };
	  //	  cout<<"This path "<<cur_dir->d_name<<endl;
	  //	  char temppath[MAX_PATHH] = { 0 }; 
	  char* temppath = new char[MAX_PATHH];
	  sprintf(temppath, "%s/%s", home, cur_dir->d_name);
	  //	  cout<<"Temp "<<temppath<<", "<<strstr(temppath,"jpg")<<endl;

	    paths.push_back(temppath);
	    i++;
	  
	}
    }
  sort(paths.begin(), paths.end(), mycompare);

  for(i=0;i<CLASSES;i++) {
    DIR* subd = opendir(paths[i]);
    struct dirent* cur_subdir;
    int count = 0;
    while((cur_subdir = readdir(subd))!=NULL)
      { 
	//std::string tmpFileName = szTempDir;
	
	if ((strcmp( cur_subdir->d_name, ".")!=0)&& (strcmp(cur_subdir->d_name, "..")!= 0)) {
	  char* file=new char[MAX_PATHH];
	  sprintf(file, "%s/%s", paths[i], cur_subdir->d_name);
	  if(strstr(file, "jpg") !=NULL ) {
	    whole_list[i].push_back(file);   
	    count++;
	  }
	}
      }
    closedir(subd);	      
    class_files[i] = count;  
  }
  closedir(d);

  //Set up train and test lists
  //Train: 30 images in each class; Test: 25 images in each class.
  std::set<int> train_indices[CLASSES];
  std::set<int> test_indices[CLASSES];
  FILE* fp1 = fopen("trainlist", "w");
  FILE* fp2 = fopen("testlist", "w");
  
  for(int i=0;i<CLASSES;i++) {
    int class_size = class_files[i];
    
    for(int it = 0;it<30;it++) {
      int rand_t = rand()%class_size;
      std::pair<std::set<int>::iterator,bool> ret = train_indices[i].insert(rand_t); 
      while(ret.second == false) {
	rand_t = rand()%class_size;
	ret = train_indices[i].insert(rand_t);
      }
      fprintf(fp1, "%i ", rand_t);
    }
    fprintf(fp1,"\n");
    for(int it = 0;it<25;it++) {
      int rand_t = rand()%class_size;
      std::pair<std::set<int>::iterator,bool> ret = test_indices[i].insert(rand_t); 
      while(ret.second == false || train_indices[i].find(rand_t)!=train_indices[i].end()) {
	if(ret.second!=false) {
	  test_indices[i].erase(rand_t);
	}
	rand_t = rand()%class_size;
	ret = test_indices[i].insert(rand_t);
      }
      fprintf(fp2, "%i ", rand_t);
    }
    fprintf(fp2, "\n");

  }
  fclose(fp1);
  fclose(fp2);
  char* trainfiles[TRAINSET];
  //Read the train set
  int it =0;
  for(i=0;i<CLASSES;i++) {
    std::set<int>::iterator iter;
    for(iter = train_indices[i].begin(); iter!=train_indices[i].end(); iter++) {
      trainfiles[it]= whole_list[i][*iter];
      it++;
    }
  }

  double start_time;
  if(argc>1 && argv[1][0] == '0') {

    //////////////////train encoder ////////////////
    //////// STEP 0: obtain sample image descriptors
    ///////////////////
    //Select 5000 samples randomly;
    srand(1);
    std::set<int> indices;
    for(int it = 0;it<TRAIN_GMM_SET;it++) {
      int rand_t = rand()%TRAINSET;
      std::pair<std::set<int>::iterator,bool> ret = indices.insert(rand_t); 
      while(ret.second == false) {
	rand_t = rand()%TRAINSET;
	ret = indices.insert(rand_t);
      }
    }

    i=0;
    std::set<int>::iterator iter;
    start_time = wallclock();
#if 1
    int count = 0;
    for(iter=indices.begin(); iter!=indices.end();iter++) {
      //  for(i=0;i<TRAIN;i++) {
    int filei=*iter;

    char imagename[256], imagesizename[256];
    /*
      sprintf(imagename, "rgbs%d", filei);
      int height = imagesizes[filei-1][0];
      int width = imagesizes[filei-1][1];
    */
    int height , width;
    float* pca_desc;
    //get descriptors
    cout<<"Train file "<<filei<<", "<<trainfiles[filei]<<endl;
    int pre_size;
    //    if(count == 2)
    pre_size = dsift_train(trainfiles[filei], &sift_res, &sift_frame, height, width);

    //Select numImages columns randomly

    cout<<"File "<<filei<<", "<<numImages<<endl;
    srand(1);
    std::set<int> indices;
    //pre_size = 2560;
    for(int it = 0;it<numImages;it++) {
      int rand_t = rand()%pre_size;
      std::pair<std::set<int>::iterator,bool> ret = indices.insert(rand_t); 
      while(ret.second == false) {
	rand_t = rand()%pre_size;
	ret = indices.insert(rand_t);
      }
    }
    std::set<int>::iterator iter;
    int it = 0;
    for(iter=indices.begin(); iter!=indices.end(); iter++) {
      for(int k=0;k<128;k++) {
	//	cout<<"It is "<<*iter<<endl;
	final_res[(numImages*i+it)*128 + k] = sift_res[(*iter)*128+k];
	/*
	if( final_res[(numImages*i+it)*128+k] > 1) {
	  cout<<*iter<<" Value "<<final_res[(numImages*i+it)*128+k]<<endl;
	  }*/
      }
      for(int k=0;k<2;k++) {
	final_frame[(numImages*i+it)*2+k] = sift_frame[(*iter)*2+k];
      }

      it++;
    }
    //cout<<"End of extraction"<<endl;
    //    free(pca_desc);
    free(sift_res);
    free(sift_frame);
    i++;
  }
#endif
  /////////////STEP 1: PCA
  const int num_variables = 128;
  const int num_records = numImages*TRAIN_GMM_SET;

  stats::pca pca(num_variables);
  //  pca.set_do_bootstrap(true, 100);

  float* projectionCenter = (float*) malloc(128*sizeof(float));
  float* projection = (float*) malloc(128*80*sizeof(float));

  int j=0;
  for (i=0; i<num_records; ++i) {
    vector<double> record(num_variables);
    for (auto value=record.begin(); value!=record.end(); ++value) {
      *value = final_res[j];
      j++;
    }
    pca.add_record(record);
  }
  pca.solve_for_vlfeat();

  const auto means1 = pca.get_mean_values();
  for(i=0;i<128;i++) {
    projectionCenter[i] = means1[i];
  }
  for(i=0;i<80;i++) {
    const auto eigenv = pca.get_eigenvector(i);
    for(j=0;j<128;j++) {
      projection[i*128+j] = eigenv[j];
    }
  }
  ///////////////////////////STEP 2  (optional): geometrically augment the features
  //////////////////////////////////////////
  //  sift_res+frames;
#define DST_DIM 80
  int pre_size = numImages*TRAIN_GMM_SET;
  float* dest=(float*)malloc(pre_size*(DST_DIM+2)*sizeof(float));
  /*
  for(i=0;i<DST_DIM;i++) {
    for(int j=0;j<pre_size;j++) {
      float sum = 0;
      for(int k=0;k<128;k++) {
	sum += projection[i*128+k]*(final_res[k+j*128] - projectionCenter[k]);
      }
      dest[i+j*(2+DST_DIM)] = sum;
    }
    }*/
    gpu_pca_mm(projection, projectionCenter, final_res, dest, pre_size, DST_DIM);
  for(i=0;i<pre_size;i++) {
    //    float halfwidth = ((float)width)/2;
    dest[i*(DST_DIM+2) +DST_DIM] = final_frame[i*2];
    dest[i*(DST_DIM+2) + DST_DIM+1] = final_frame[i*2+1];
  }
  cout<<"End of step 2"<<endl;
  //////////////////////STEP 3  learn a GMM vocabulary
  //////////////////////////
  numData = pre_size;
  dimension = 82;
  numComponents = 256;
  //vl_twister
  VlRand * rand;
  rand = vl_get_rand();
  vl_rand_seed(rand, 1);

  VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numComponents);
  ///////////////////////WARNING: should set these parameters
    vl_gmm_set_initialization (gmm, VlGMMKMeans) ;
    //Compute V
    double denom = pre_size-1;
    double xbar[82], V[82];
    for(i=0;i<dimension;i++) {
      xbar[i] = 0.0;
      for(int j=0;j<numData;j++) {
	xbar[i] += (double)dest[j*dimension + i];
      }
      xbar[i] /= (double)pre_size;
    }
    for(i=0;i<dimension;i++) {
      double absx = 0.0;
      for(int j=0;j<numData;j++) {
        double tempx = (double)dest[j*dimension + i] - xbar[i];
	absx += abs(tempx) * abs(tempx);
      }
      V[i] = absx/denom;
    }

    //Get max(V)
    double maxNum = V[0];
    for(i=1;i<dimension;i++) {
      if(V[i] > maxNum) {
	maxNum = V[i];
      }
    }
    maxNum = maxNum * 0.0001;
    vl_gmm_set_covariance_lower_bound (gmm, (double)maxNum);
    cout<<"Lower bound "<<maxNum<<endl;
    vl_gmm_set_verbosity(gmm, 1);
    vl_gmm_set_max_num_iterations(gmm, 100);
    //    printf("vl_gmm: initialization = %s\n", initializationName) ;
    printf("vl_gmm: maxNumIterations = %d\n", vl_gmm_get_max_num_iterations(gmm)) ;
    printf("vl_gmm: numRepetitions = %d\n", vl_gmm_get_num_repetitions(gmm)) ;
    printf("vl_gmm: data type = %s\n", vl_get_type_name(vl_gmm_get_data_type(gmm))) ;
    printf("vl_gmm: data dimension = %d\n", dimension) ;
    printf("vl_gmm: num. data points = %d\n", numData) ;
    printf("vl_gmm: num. Gaussian modes = %d\n", numClusters) ;
    printf("vl_gmm: lower bound on covariance = [") ;
    printf(" %f %f ... %f]\n",
                vl_gmm_get_covariance_lower_bounds(gmm)[0],
                vl_gmm_get_covariance_lower_bounds(gmm)[1],
                vl_gmm_get_covariance_lower_bounds(gmm)[dimension-1]) ;
    //  VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numComponents);
  
    double gmmres = vl_gmm_cluster(gmm, dest, numData);
    cout<<"GMM ending cluster."<<endl;
  
    priors = (TYPE*)vl_gmm_get_priors(gmm);
    means = (TYPE*)vl_gmm_get_means(gmm);
    covariances = (TYPE*)vl_gmm_get_covariances(gmm);
    cout<<"End of encoder "<<endl;  
    cout<<"Training time "<<wallclock() - start_time<<endl;
    ///////////////END train encoer//////////

  //output priors
    ofstream fp("cluster");
    if(!fp) {
      cout<<"Output file wrong\n";
    }
  /*    TYPE* priors = (TYPE*)vl_gmm_get_priors(gmm);
	TYPE* means = (TYPE*)vl_gmm_get_means(gmm);
	TYPE* covariances = (TYPE*)vl_gmm_get_covariances(gmm);
  */
  //output means
    for(i=0;i<dimension;i++) {
      for (int j=0;j<numClusters;j++) {
	fp<<means[i + dimension * j]<<" ";
      }
    }
    fp<<endl;
    //output covariances
    for(i=0;i<dimension;i++) {
      for (int j=0;j<numClusters;j++) {
	fp<<covariances[i+dimension * j]<<" ";
      }
    }
    fp<<endl;
    for(i=0;i<numClusters;i++) {
      fp<<priors[i]<<" ";
      cout<<i<<" Priors "<<priors[i]<<endl;
    }
    fp<<endl;
    fp.close();
    
    //////////output projection
    ofstream center("center");
    if(!center) {
      cout<<"Output file wrong\n";
    }
    for(i=0;i<DST_DIM;i++) {
      for(int j=0;j<128;j++) {
	center<<projection[i+j*DST_DIM]<<' ';
      }
    }
    for(i=0;i<128;i++) {
      center<<projectionCenter[i]<<' ';
    }
    center.close();
  }
  else {
    //Process the images (with the encoder )
    char* testfiles[TESTSET];
    //Read the test set
    int it =0;
    for(i=0;i<CLASSES;i++) {
      std::set<int>::iterator iter;
      int count =0;
      for(iter = test_indices[i].begin(); iter!=test_indices[i].end(); iter++) {
	testfiles[it]= whole_list[i][*iter];
	it++;
	count++;
      }
    }
    cout<<"Total test files "<<it<<endl;
    TYPE* encs_train = new TYPE[TRAINSET*SIZE];
    TYPE** encs_test = new TYPE*[TESTSET];
    int num, j= 0;
    char name1[10];

    //Read the parameters
    priors = (TYPE*) vl_malloc(sizeof(float)*numClusters);
    means = (TYPE*) vl_malloc(sizeof(float)*dimension*numClusters);
    covariances = (TYPE*) vl_malloc(sizeof(float)*dimension*numClusters);
    
    ifstream fp("cluster");
    if(!fp) {
      cout<<"Input file wrong\n";
    }
    //read priors
    string temp;

    getline(fp, temp);
    stringstream line1(temp);
    //read means
    for(i=0;i<dimension;i++) {
      for (int j=0;j<numClusters;j++) {
	line1>>means[i+dimension * j];
      }
    }
    //read covariances
    getline(fp, temp);
    stringstream line2(temp);
    for(i=0;i<dimension;i++) {
      for (int j=0;j<numClusters;j++) {
	line2>>covariances[i+dimension * j];
      }
    }    
    getline(fp, temp);
    stringstream line3(temp);
    for(i=0;i<numClusters;i++) {
      line3>>priors[i];
    }

    float* projection=(float*)malloc(128*sizeof(float)*80);
    float* projectionCenter=(float*)malloc(128*sizeof(float));

    ifstream center("center");
    for(i=0;i<DST_DIM;i++) {
      for(int j=0;j<128;j++) {
	center>>projection[i+j*DST_DIM];
      }
    }
    for(i=0;i<128;i++) {
      center>>projectionCenter[i];
    }
    center.close();

    start_time = wallclock();
    dsift_init();
    gpu_copy(covariances, priors, means, numClusters, dimension);
    //Encode train files
    char tempname[16];
#if 1
    for(i=0;i<TRAINSET;i++) 
      {
       	sprintf(tempname, "C_output/R%d", i+1);
	//	encs_train[i] = (TYPE*)vl_malloc(sizeof(float)*2 * dimension * numClusters);
	//	if(i>=297)
	char tempname1[256];
	//sprintf(tempname1, "/home/wenjing/vlfeat/myexperiment/test_video/frames_baby_1/frame0.jpg");
	//	sprintf(tempname1, "/home/wenjing/vlfeat/apps/recognition/data/caltech256/256_ObjectCategories/013.birdbath/013_0087.jpg");
	//	if(i>1200)
       	onlineProcessing(trainfiles[i], priors, means, covariances, &encs_train[i*SIZE], projection, projectionCenter, tempname);
      }
#endif
    cout<<endl<<"----------------"<<argc<<endl;
    cout<<endl;

    //Encode test files


#if 1
    //    encs_test[0] = (TYPE*)vl_malloc(sizeof(float)*2 * dimension * numClusters);
    for(i=0;i<TESTSET;i++) 
      {
	//	char temp[3];
	//	sprintf(temp, "%d", i+1);
	encs_test[i] = (TYPE*)vl_malloc(sizeof(float)*2 * dimension * numClusters);
	sprintf(tempname, "C_output/E%d", i+1);
	//	cout<<"testfiles[i] "<<testfiles[i]<<endl;
	char tempname1[256];
	//sprintf(tempname1, "/media/sdb1/mawenjing/yulei/vlfeat/data/caltech256/256_ObjectCategories/030.canoe/030_0097.jpg");
	//onlineProcessing(tempname1, priors, means, covariances, encs_test[i], projection, projectionCenter, tempname);
       	onlineProcessing(testfiles[i], priors, means, covariances, encs_test[i], projection, projectionCenter, tempname);
      }
#endif    
    gpu_free();
    dsift_finalize();
    free(projection);
    free(projectionCenter);
    free(priors);
    free(means);
    free(covariances);
    cout<<"Encoding time "<<wallclock() - start_time<<", including "<<file_time<<". So it is "<<wallclock() - start_time - file_time<<endl;
    ////////////////////////////////////
    ////////////////////////svm classification/////////
    /////////////////////////////////////////
    /*
      input: class_members (20 files)    
    */
    start_time = wallclock();
    double myap = 0.0;
    int begin = 0, end = 0;
    double desc[TRAINSET];
    int desc_test[TESTSET];
    int difficult[TESTSET];
    double** vlW = new double*[CLASSES]; 
    double* vlb = new double[CLASSES];
    //    memset(desc_test, 0, TESTSET*sizeof(double));
    for(int it=0;it<TESTSET;it++) 
      desc_test[i] = -1;

    memset(difficult, 0, TESTSET*sizeof(int));
    int* preds = new int[TESTSET];
    double** scores = new double*[CLASSES];
      ofstream svmfile("svmWb");
      if(!svmfile) {
	cout<<"Output file wrong\n";
      }

    for(i=0;i<CLASSES;i++) {
      char classname[256];
      scores[i] = new double[TESTSET];
      vlW[i] = new double[SIZE];
      /*
	getline(classfile, temp);
	stringstream line(temp);
      */
      //Get the labels of the training set for this class
      //while(getline(classfile, temp)) {
#if 1
      int it =0;
      for(int ini=0;ini<CLASSES;ini++) {
	std::set<int>::iterator iter;
	
	for(iter = train_indices[ini].begin(); iter!=train_indices[ini].end(); iter++) {
	  if(ini == i) {
	    desc[it]= 1;//whole_list[i][*iter];
	  }
	  else {
	    desc[it] = -1;
	  }
	  it++;
	}
      }

      //Do svm training
      //      train(encs_train, desc);
      double lambda = 1/(10.0*(double)TRAINSET);
      cout<<"lambda is "<<lambda<<endl;
      //Using the svm train function in Vlfeat
      vlsvmtrain(vlW[i], &vlb[i], encs_train, desc, TRAINSET, lambda);
      
      for(int j=0;j<SIZE;j++) {
	svmfile<<vlW[i][j]<<' ';
      }
      svmfile<<vlb[i]<<"\n";
      
      //Load the labels and calculate the difficult values 
      //Get the labels of the testing set for this class
      start = wallclock();
      //      delete encs_train;
#endif
    }
    svmfile.close();

    //Load vlW and vlb
    /*
    ifstream svm1("svmWb");
    for(i=0;i<CLASSES;i++) {
      for(int j=0;j<SIZE;j++) {
	svm1>>vlW[i][j];
      }
      svm1>>vlb[i];
    }
    svm1.close();
    */
    calc_score(scores, encs_test, desc_test, vlW, vlb, preds);
    for(i=0;i<CLASSES;i++) {
      delete vlW[i];
    }
    delete vlW;
    delete vlb;

    //For calculation of accuracy, we need scores from all classes
    for(i=0;i<CLASSES;i++) {
      it =0;
      for(int ini=0;ini<CLASSES;ini++) {
	std::set<int>::iterator iter;
	for(iter = test_indices[ini].begin(); iter!=test_indices[ini].end(); iter++) {
	  if(ini == i) {
	    desc_test[it]= 1;//whole_list[i][*iter];
	  }
	  else {
	    desc_test[it] = -1;
	  }
	  //	  cout<<"class "<<i<<" desc "<<it<<" is "<<desc_test[it]<<endl;
	  difficult[it] = 0;
	  it++;
	}
      }

      //Do prediction
      /*
      std::set<int>::iterator iter;
      for(iter = test_indices[i].begin();iter!=test_indices[i].end();iter++) {
	desc_test[begin+it] = 1;
	it++;
      }
      end = begin+it;
      */
      //      cout<<"Label time "<<wallclock() - start<<endl;
      //Load encs_test from the files
      start = wallclock();

      myap += my_do_predict_acc(i, preds, desc_test); //encs_test, desc_test, difficult, vlW, vlb, i);
    }
    
    cout<<"Avg AP "<<myap/256<<endl;
  }
  cout<<"Classification time "<<wallclock() - start_time<<endl;
  return 0;
}
