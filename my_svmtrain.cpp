#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
//#include "linear.h"
#include <iostream>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#include <vector>
#include <algorithm>
#include "vl/svm.h"
#include "vl/mathop.h"
#include "cuda_files.h"

using namespace std;
/*struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
*/
int flag_cross_validation;
int nr_fold;
double bias;
//static char *line = NULL;
static int max_line_len;

#define SIZE 41984
#define TESTSET 6400
#define TRAINSET 7680
#define CLASSES 256


struct labelpair {
  int label;
  double score;
};
bool compare_as_ints (struct labelpair i, struct labelpair j)
{
  return (i.score>j.score);
}

void vlsvmtrain(double* vlW, double* vlb, float* indata, double* labels, int numData, float lambda) {
  enum {IN_DATASET = 0, IN_LABELS, IN_LAMBDA, IN_END} ;
  enum {OUT_MODEL = 0, OUT_BIAS, OUT_INFO, OUT_SCORES, OUT_END} ;

  vl_int opt, next;
  //  mxArray const *optarg ;

  VlSvmSolverType solver = VlSvmSolverSdca ;
  VlSvmLossType loss = VlSvmLossHinge ;
  int verbose = 1;
  VlSvmDataset * dataset ;
  //  double * labels ;
  double * weights = NULL ;
  //  double lambda ;

  double epsilon = 0.001 ;
  double biasMultipler = 1 ;
  vl_index maxNumIterations = TRAINSET*100 ;
  vl_index diagnosticFrequency = TRAINSET ;
  //mxArray const * matlabDiagnosticFunctionHandle = NULL ;

  //  mxArray const * initialModel_array = NULL ;
  double initialBias = VL_NAN_D ;
  vl_index startingIteration = -1 ;

  /* SGD */
  double sgdBiasLearningRate = -1 ;

  //  VL_USE_MATLAB_ENV ;

  /* Mode 1: pass data, labels, lambda, and options */
  //  mxArray const* samples_array = indata;
  vl_size dimension ;
  vl_size numSamples ;
  void * data ;
  vl_type dataType ;

  dataType = VL_TYPE_FLOAT;
  data = indata; //mxGetData(samples_array) ;
  dimension = SIZE; //mxGetM(samples_array) ;
  numSamples = numData; //mxGetN(samples_array) ;
  dataset = vl_svmdataset_new(dataType, data, dimension, numSamples) ;
  //  diagnosticFrequency = 15;
  {
    VlSvm * svm = vl_svm_new_with_dataset(solver, dataset, labels, lambda) ;
    //DiagnosticOpts dopts ;

    if (epsilon >= 0) vl_svm_set_epsilon(svm, epsilon) ;
    if (maxNumIterations >= 0) vl_svm_set_max_num_iterations(svm, maxNumIterations) ;
    if (biasMultipler >= 0) vl_svm_set_bias_multiplier(svm, biasMultipler) ;
    if (sgdBiasLearningRate >= 0) vl_svm_set_bias_learning_rate(svm, sgdBiasLearningRate) ;
    if (diagnosticFrequency >= 0) vl_svm_set_diagnostic_frequency(svm, diagnosticFrequency) ;
    if (startingIteration >= 0) vl_svm_set_iteration_number(svm, (unsigned)startingIteration) ;
    if (weights) vl_svm_set_weights(svm, weights) ;
    vl_svm_set_loss (svm, loss) ;
    /*
    dopts.verbose = verbose ;
    dopts.matlabDiagonsticFunctionHandle = NULL; //matlabDiagnosticFunctionHandle ;
    vl_svm_set_diagnostic_function (svm, (VlSvmDiagnosticFunction)diagnostic, &dopts) ;
    */

    if (verbose) {
      double C = 1.0 / (vl_svm_get_lambda(svm) * vl_svm_get_num_data (svm)) ;
      char const * lossName = 0 ;
      switch (loss) {
        case VlSvmLossHinge: lossName = "hinge" ; break ;
        case VlSvmLossHinge2: lossName = "hinge2" ; break ;
        case VlSvmLossL1: lossName = "l1" ; break ;
        case VlSvmLossL2: lossName = "l2" ; break ;
        case VlSvmLossLogistic: lossName = "logistic" ; break ;
      }
      printf("vl_svmtrain: parameters (verbosity: %d)\n", verbose) ;
      printf("\tdata dimension: %d\n",vl_svmdataset_get_dimension(dataset)) ;
      printf("\tnum samples: %d\n", vl_svmdataset_get_num_data(dataset)) ;
      printf("\tlambda: %g (C equivalent: %g)\n", vl_svm_get_lambda(svm), C) ;
      printf("\tloss function: %s\n", lossName) ;
      printf("\tmax num iterations: %d\n", vl_svm_get_max_num_iterations(svm)) ;
      printf("\tepsilon: %g\n", vl_svm_get_epsilon(svm)) ;
      printf("\tdiagnostic frequency: %d\n", vl_svm_get_diagnostic_frequency(svm)) ;
      printf("\tusing custom weights: %s\n", VL_YESNO(weights)) ;
      printf("\tbias multiplier: %g\n", vl_svm_get_bias_multiplier(svm)) ;
      switch (vl_svm_get_solver(svm)) {
        case VlSvmSolverNone:
          printf("\tsolver: none (evaluation mode)\n") ;
          break ;
        case VlSvmSolverSgd:
          printf("\tsolver: sgd\n") ;
          printf("\tbias learning rate: %g\n", vl_svm_get_bias_learning_rate(svm)) ;
          break ;
        case VlSvmSolverSdca:
          printf("\tsolver: sdca\n") ;
          break ;
      }
    }

    cout<<"Start training"<<endl;
    double start = wallclock();
    vl_svm_train(svm) ;
    cout<<"Time "<<wallclock() - start<<endl;
    /*
    {
      mwSize dims[2] ;
      dims[0] = vl_svmdataset_get_dimension(dataset) ;
      dims[1] = 1 ;
      out[OUT_MODEL] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL) ;
      memcpy(mxGetPr(out[OUT_MODEL]),
             vl_svm_get_model(svm),
             vl_svm_get_dimension(svm) * sizeof(double)) ;
    }
    */
    memcpy(vlW, vl_svm_get_model(svm), dimension*sizeof(double));
    cout<<"vlW "<<vlW[0]<<", "<<*vlb<<endl;
    /*
    for(int i=0;i<dimension;i++) {
      if(i<1000 && vlW[i]!=0) {
	cout<<i<<", "<<vlW[i]<<endl;
      }
      }*/
    *vlb = vl_svm_get_bias(svm);
    /*
    out[OUT_BIAS] = vlmxCreatePlainScalar(vl_svm_get_bias(svm)) ;
    if (nout >= 3) {
      out[OUT_INFO] = makeInfoStruct(svm) ;
    }
    if (nout >= 4) {
      mwSize dims[2] ;
      dims[0] = 1 ;
      dims[1] = vl_svmdataset_get_num_data(dataset) ;
      out[OUT_SCORES] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL) ;
      memcpy(mxGetPr(out[OUT_SCORES]),
             vl_svm_get_scores(svm),
             vl_svm_get_num_data(svm) * sizeof(double)) ;
    }

    */
    vl_svm_delete(svm) ;
    /*
    if (vl_svmdataset_get_homogeneous_kernel_map(dataset)) {
      VlHomogeneousKernelMap * hom = vl_svmdataset_get_homogeneous_kernel_map(dataset) ;
      vl_svmdataset_set_homogeneous_kernel_map(dataset,0) ;
      vl_homogeneouskernelmap_delete(hom) ;
      }*/
    vl_svmdataset_delete(dataset) ;

  }
}
double my_do_predict(float** encs, double* labels, int* difficult, double* vlW, double vlb, int class_index) {
  ///////////Do prediction
  int correct = 0;
  int total = 0;
  double error = 0;
  double sump = 0, sumn = 0, sumpp = 0, sumtt = 0, sumpt = 0;
  struct feature_node *x;
  struct labelpair predict_label[TESTSET];
  /*
  if((model_=load_model("model"))==0)
    {
      exit(1);
    }
  */
  int nr_class=2; //get_nr_class(model_);
  double *prob_estimates=NULL;
  int j, n;
  double start;
  int nr_feature=SIZE; //get_nr_feature(model_);
  cout<<"Features "<<nr_feature<<endl;
  //  cout<<"Bias "<<model_->bias<<endl;

  int tp[TESTSET+1], fp[TESTSET+1];
  tp[0]=0;
  fp[0]=0;
  cout<<"tp "<<tp[0]<<endl;
  //Don't predict probability for now

  int max_nr_attr = SIZE+2;
  //x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
    start = wallclock();
  for(int ins=0;ins<TESTSET;ins++) {
    double target_label;
    char *idx, *val, *endptr;
    int inst_max_index = 0; // strtol gives 0 if wrong format
    //    cout<<"See predict"<<endl;
    predict_label[ins].label = labels[ins] - difficult[ins];
    //    cout<<"Label "<<labels[ins]<<", "<<difficult[ins]<<endl;
    /*
    for(int j=0;j<SIZE;j++) {
      //read each value of the fisher vector
      x[j].index = j;
      x[j].value = enc[ins][j];
    }
    //No bias for now
    x[SIZE].index = -1;
    */
    //////////////////////
    //////////////my predict//////////
    /////////////////

    double tmp = 0;
    for(int i=0;i<SIZE;i++) {
      tmp += encs[ins][i]*vlW[i];
    }
    predict_label[ins].score = tmp + vlb; //my_predict(vlW, vlb ,x);
    //    cout<<"Score is "<<predict_label[ins].score<<endl;
    /*
    if(predict_label == target_label)
      ++correct;
    error += (predict_label-target_label)*(predict_label-target_label);
    */

    /*
    sumt += target_label;
    sumpp += predict_label*predict_label;
    sumtt += target_label*target_label;
    sumpt += predict_label*target_label;
    ++total;
    */
  }
  cout<<"Score time "<<wallclock() - start<<endl;
  /////////////////////////////
  ////////Do the calculations in MATLAB code
  ///////////////////////////////
  //TODO: sort label pairs
  start = wallclock();
  std::vector<labelpair> myvector(predict_label, predict_label+sizeof(predict_label)/sizeof(labelpair));
  //  myvector.assign(predict_label);//mydoubles,mydoubles+8);
  std::stable_sort(myvector.begin(), myvector.end(), compare_as_ints);
  cout<<"Sort "<<wallclock() - start<<endl;

  start = wallclock();
  for(int ins=0;ins<TESTSET;ins++) {
    tp[ins+1] = tp[ins];
    fp[ins+1] = fp[ins];
    cout<<"sorted "<<myvector[ins].label<<", "<<myvector[ins].score<<endl;
    if(myvector[ins].label>0) {
      sump ++;
      tp[ins+1]++;
    }
    else if(myvector[ins].label<0){
      sumn ++;
      fp[ins+1]++;
    }
    //    cout<<"label and tp "<<predict_label[ins].label<<" "<<tp[ins]<<", "<<tp[ins+1]<<endl;
  }

#define SMALL 1e-10
  double recall[TESTSET+1], precision[TESTSET+1];
  for(int ins=0;ins<TESTSET+1;ins++) {
    recall[ins] = tp[ins]/std::max(SMALL, ((double)sump));
    //    cout<<"Tp and recall "<<tp[ins]<<", "<<recall[ins]<<endl;
  }
  for(int ins=0;ins<TESTSET+1;ins++) {
    precision[ins] = max(((double)tp[ins]), SMALL) / max(SMALL, ((double)tp[ins]+fp[ins]));
  }
  double sumprec = 0.0;
  for(int ins=1;ins<TESTSET+1;ins++) {
    double diff = recall[ins]-recall[ins-1];
    if(diff!=0){
      sumprec += precision[ins];
      cout<<"predision "<<precision[ins]<<endl;
    }
  }
  cout<<class_index<<" AP is "<<sumprec/sump<<endl;
  cout<<"Other time "<<wallclock() - start<<endl;
  return sumprec/sump;
  ////////////////////
  //  free_and_destroy_model(&model_);

  
  //  destroy_param(&param);
  //  free(prob.y);
  //  free(prob.x);
  //  free(x_space);
  
  //  free(line);

}

double my_do_predict_acc(float** encs, double* labels, int* difficult, double* vlW, double vlb, int class_index) {
  ///////////Do prediction
  int correct = 0;
  int total = 0;
  double error = 0;
  double sump = 0, sumn = 0, sumpp = 0, sumtt = 0, sumpt = 0;
  struct feature_node *x;
  struct labelpair predict_label[TESTSET];
  /*
  if((model_=load_model("model"))==0)
    {
      exit(1);
    }
  */
  int nr_class=2; //get_nr_class(model_);
  double *prob_estimates=NULL;
  int j, n;
  double start;
  int nr_feature=SIZE; //get_nr_feature(model_);
  cout<<"Features "<<nr_feature<<endl;
  //  cout<<"Bias "<<model_->bias<<endl;

  int tp[TESTSET+1], fp[TESTSET+1];
  tp[0]=0;
  fp[0]=0;
  cout<<"tp "<<tp[0]<<endl;
  //Don't predict probability for now

  int max_nr_attr = SIZE+2;
  //x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
    start = wallclock();
  for(int ins=0;ins<TESTSET;ins++) {
    double target_label;
    char *idx, *val, *endptr;
    int inst_max_index = 0; // strtol gives 0 if wrong format
    //    cout<<"See predict"<<endl;
    predict_label[ins].label = labels[ins] - difficult[ins];
    //    cout<<"Label "<<labels[ins]<<", "<<difficult[ins]<<endl;
    /*
    for(int j=0;j<SIZE;j++) {
      //read each value of the fisher vector
      x[j].index = j;
      x[j].value = enc[ins][j];
    }
    //No bias for now
    x[SIZE].index = -1;
    */
    //////////////////////
    //////////////my predict//////////
    /////////////////

    double tmp = 0;
    for(int i=0;i<SIZE;i++) {
      tmp += encs[ins][i]*vlW[i];
    }
    predict_label[ins].score = tmp + vlb; //my_predict(vlW, vlb ,x);
    cout<<ins<<" Score is "<<predict_label[ins].score<<", label "<<labels[ins]<<endl;
  }
  cout<<"Score time "<<wallclock() - start<<endl;
}

void calc_score(double** scores, float** encs, int* labels, double** vlW, double* vlb, int* preds) {
  /////////////////////////////
  ////////Do the calculations in MATLAB code
  ///////////////////////////////
  int i;
  //  double** scores = new double*[CLASSES];
  //Calculate scores
  for(i=0;i<CLASSES;i++) {
    //scores[i] = new double[TESTSET];
    for(int ins=0;ins<TESTSET;ins++) {
      double tmp = 0;
      for(int j=0;j<SIZE;j++) {
	tmp += encs[ins][j]*vlW[i][j];
      }
      scores[i][ins] = tmp + vlb[i]; //my_predict(vlW, vlb ,x);
    }
  }
  //Find preds
  double* temp_score = new double[TESTSET];
  for(i=0;i<TESTSET;i++) {
    temp_score[i] = -2000;
    preds[i] = 0;
    for(int j=0;j<CLASSES;j++) {
      cout<<"Score is "<<scores[j][i]<<endl;
      if(scores[j][i]>temp_score[i]) {
	temp_score[i] = scores[j][i];
	preds[i] = j;
      }
    }
  }
  delete temp_score;
}
double my_do_predict_acc(int class_index, int* preds, int* labels) {
  //  for(i=0;i<CLASSES;i++) {
  float tmp = 0;
    //}
  int i;
  for(i=0;i<TESTSET;i++) {
    //    for(int j=0;j<CLASSES;j++) {
    if(labels[i] == 1) {
      //Count this preds[i]
      cout<<"Non zero "<<class_index<<", "<<preds[i]<<endl;
      if(preds[i] == class_index) {
	tmp += 1;
      }
    }
  }
  cout<<class_index+1<<" ACC is "<<tmp<<", "<<tmp/25<<endl;
  //TODO: it should not be fixed
  return tmp/25;
}


#if 0
  //TODO: sort label pairs
  start = wallclock();
  //  std::vector<labelpair> myvector(predict_label, predict_label+sizeof(predict_label)/sizeof(labelpair));
  //  myvector.assign(predict_label);//mydoubles,mydoubles+8);
  //  std::stable_sort(myvector.begin(), myvector.end(), compare_as_ints);
  //  cout<<"Sort "<<wallclock() - start<<endl;

  start = wallclock();
  double good = 0.0;
  for(int ins=0;ins<TESTSET;ins++) {
    if(predict_label[ins].score>0 && labels[ins]>0)
      good ++;
    else     if(predict_label[ins].score<0 && labels[ins]<0)
      good ++;
  }
  double acc = good/TESTSET;
  cout<<class_index<<", acc is "<<acc<<endl;
  return 0;

#define SMALL 1e-10
  double recall[TESTSET+1], precision[TESTSET+1];
  for(int ins=0;ins<TESTSET+1;ins++) {
    recall[ins] = tp[ins]/std::max(SMALL, ((double)sump));
    //    cout<<"Tp and recall "<<tp[ins]<<", "<<recall[ins]<<endl;
  }
  for(int ins=0;ins<TESTSET+1;ins++) {
    precision[ins] = max(((double)tp[ins]), SMALL) / max(SMALL, ((double)tp[ins]+fp[ins]));
  }
  double sumprec = 0.0;
  for(int ins=1;ins<TESTSET+1;ins++) {
    double diff = recall[ins]-recall[ins-1];
    if(diff!=0){
      sumprec += precision[ins];
      cout<<"predision "<<precision[ins]<<endl;
    }
  }
  cout<<class_index<<" AP is "<<sumprec/sump<<endl;
  cout<<"Other time "<<wallclock() - start<<endl;
  return sumprec/sump;
  ////////////////////
  //  free_and_destroy_model(&model_);

  //  destroy_param(&param);
  //  free(prob.y);
  //  free(prob.x);
  //  free(x_space);
  
  //  free(line);

}
#endif
