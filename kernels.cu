#define ITEMS 4
#define ITEMS3 192
#define THREADS 256
#define THREADS2 96
#define THREADS3 96
#define TYPE float
#define TILE 4
#define THREADS_S 128

#include "/usr/local/cuda/samples/common/inc/helper_math.h"
//#define VL_INFINITY_D 0x7FF0000000000000ui64
#define VL_MAX(x,y) (((x)>(y))?(x):(y))

#define VL_GMM_MIN_VARIANCE 1e-6
#define VL_GMM_MIN_POSTERIOR 1e-2
#define VL_GMM_MIN_PRIOR 1e-6
//__constant float priors_d[1024];
float *src_d, *tmp1_d, *tmp2_d, *image_d, *grads_d, *priors_d, *means_d, *posteriors_d, *sqrtInvSigma_d;
TYPE *covariances_d, *logWeights_d, *logCovariances_d, *invCovariances_d, *enc_d, *sum_d;
int *goout_d;

__global__ void gmm_1(float* covariances, float* invCovariances, float* logCovariances, float* logWeights, float* priors, int dimension, int numClusters, float infinity, float* sqrtInvSigma) {
  //  int tid = get_local_id(0);
  int dim = 0;
  int i_cl = blockIdx.x*THREADS + threadIdx.x; //get_global_id(0);
  float logSigma = 0;

  if(priors[i_cl] < VL_GMM_MIN_PRIOR) {
    logWeights[i_cl] = infinity; // -(TYPE) VL_INFINITY_D;
  }
  else {
    logWeights[i_cl] = log(priors[i_cl]);
  }
  for(dim = 0; dim<dimension; ++dim) {
    logSigma += log(covariances[i_cl*dimension + dim]);
    float temp = (float) 1.0/ covariances[i_cl*dimension + dim];
    invCovariances[i_cl*dimension + dim ] = temp;
    sqrtInvSigma[i_cl*dimension + dim] = sqrt(temp);
  }
  
  logCovariances[i_cl] = logSigma;
}
#define VEC 1
#define DIM 82
__global__ void gmm_2(float* invCovariances, float* logCovariances, float* logWeights, float* posteriors, int numClusters, float halfDimLog2Pi, float* data, float const* means, float infinity, int numData) {
  int tid = threadIdx.x;
  int gid = blockIdx.x;// * THREADS2 + threadIdx.x; //get_group_id(0);

  __shared__ float temp[THREADS2 * TILE]; //2K
  __shared__ float data_l[THREADS2*TILE]; //1.6K

  //  __shared__ float cluster_l[THREADS2 * TILE]; //2K
  __shared__ float maxPosterior[ TILE];
  //local memory 512*4*4B, 8K
  float p, p2, p1;

  //  float mydata[128];
  int i_cl = 0, i=0;
  //  float tt[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
  float tt0=0.0,  tt1=0.0,  tt2=0.0,  tt3=0.0,  tt4=0.0,  tt5=0.0,  tt6=0.0,  tt7=0.0;
  for(int gtid = gid; gtid*ITEMS<numData; gtid += gridDim.x) 
  for(int gtid_l  = gtid*ITEMS; gtid_l<(gtid+1)*ITEMS && gtid_l<numData; gtid_l+= TILE) 
    {
      for(int j = tid; j<DIM; j += THREADS2) 
	
	if(j < DIM) {

	  
	  data_l[ j] = data[(gtid_l) * DIM + j];
	  if(gtid_l + 3<numData) {
	    data_l[THREADS2 + j] = data[(gtid_l+1) * DIM + j];
	    data_l[THREADS2*2+ j] = data[(gtid_l+2) * DIM + j];
	    data_l[THREADS2*3+ j] = data[(gtid_l+3) * DIM + j];
	  }
	  else if(gtid_l + 2<numData) {
	    data_l[THREADS2+ j] = data[(gtid_l+1) * DIM + j];
	    data_l[THREADS2*2+ j] = data[(gtid_l+2) * DIM + j];
	  }
	  else if(gtid_l + 1 < numData) {
	    data_l[THREADS2+ j] = data[(gtid_l+1) * DIM + j];
	  }
	  
	}
	
      if(tid < TILE) 
      //for(i=0;i<TILE;i++)
      {
	maxPosterior[tid] = infinity; //(float)(-VL_INFINITY_D);
      }
      __syncthreads();
      //Parallelization on numClusters
      int half = THREADS2/2;
      for(i_cl = 0; i_cl < (signed)numClusters; i_cl += 2) {
	if(tid>=THREADS2/2) {
	  for(i=0;i< TILE;i++) {
	    temp[half* i + tid -half] = 0.0 ;
	    temp[half* i + tid+ half*3] = 0.0 ;
	  }
	}
	else {
	  tt0=0.0,  tt1=0.0,  tt2=0.0,  tt3=0.0,  tt4=0.0,  tt5=0.0,  tt6=0.0, tt7=0.0;
	}
	/*
	for(int i_t = 0;i_t<TILE;i_t++) {
	  float temp1  = 0.0;

	  for(int i_d = 0;i_d<DIM; i_d++) {
	    float meansl = means_s[i_d]; // means[i_cl*DIM + i_d];
	    float invl = invl_s[i_d]; // *(invCovariances + i_cl * DIM + i_d);
	    float d1 = data_l[i_d+128*i_t]-meansl;
	    temp1 += d1*d1*invl;
	  }
	  if(gtid_l+i_t<numData) {
	    float max = maxPosterior[i_t*THREADS2 + tid];
	    float p2 = logWeights[i_cl] - halfDimLog2Pi - 0.5 * logCovariances[i_cl]; 
	    float p1 = p2 - 0.5*temp1;
	    posteriors[i_cl + (gtid_l + i_t)*numClusters] = p1;
	    if(p1 > max) 
	      maxPosterior[i_t*THREADS2 + tid] = p1;
	  }
	}
	*/
	for(int j = tid; j < DIM; j +=THREADS2) {
	  if(j<DIM) {
	      float meansl = means[i_cl*DIM + j];
	      float invl = *(invCovariances + i_cl * DIM + j);
	      float meansl1 = means[(i_cl+1)*DIM + j];
	      float invl1 = *(invCovariances + (i_cl+1) * DIM + j);
	  
		if(tid>=half) {

		  tid -= half;
#if VEC
		  
		  float d1,d;
		  
		  if(gtid_l+3<numData) {
		  
		  float4 data4=make_float4(data_l[j], data_l[THREADS2+j], data_l[THREADS2*2+j], data_l[THREADS2*3+j]);
		  //		  float4 invl4 = make_float4(invl, invl, invl, invl);
		  float4 temp4=make_float4(data4.x-meansl, data4.y-meansl, data4.z-meansl, data4.w-meansl);
		  float4 temp24 = temp4*temp4;
		  temp24 = temp24 * invl;
		  temp[tid] = temp24.x;
		  temp[half+tid] = temp24.y;
		  temp[THREADS2+tid] = temp24.z;
		  temp[half*3+tid] = temp24.w;
		  ////////////////
		  temp4=make_float4(data4.x-meansl1, data4.y-meansl1, data4.z-meansl1, data4.w-meansl1);
		  temp24 = temp4*temp4* invl1;
		  temp[tid+half*4] = temp24.x;
		  temp[half*5+tid] = temp24.y;
		  temp[THREADS2*3+tid] = temp24.z;
		  temp[half*7+tid] = temp24.w;
		  } 
		  
#else
		  TYPE d1 = data_l[j];
      		  TYPE d = d1 - meansl; //[i_cl*DIM + tid + j];
		  temp[tid] = d*d* invl; //(*(invCovariances + i_cl * DIM + tid + j));
		  d = d1 - meansl1;
		  temp[tid + half*4] = d*d*invl1;
		  
		  if(gtid_l + 3 < numData) 
		  {
		    d1 = data_l[THREADS2+j];
		    d = d1 - meansl; //[i_cl*DIM + tid + j];
		    temp[half+tid] = d*d* invl;
		    d = d1 - meansl1; //[i_cl*DIM + tid + j];
		    temp[half*5+tid] = d*d* invl1;

		    d1 = data_l[THREADS2*2+j];
		    d = d1 - meansl;
		    temp[THREADS2+tid] = d*d* invl; 
		    d = d1 - meansl1;
		    temp[half*6+tid] = d*d* invl1;

		    d = data_l[half*6 + j ] - meansl; //[i_cl*DIM + tid + j];
		    temp[half*3 + tid] = d*d* invl; 
		    d = data_l[half*6 + j ] - meansl1; //[i_cl*DIM + tid + j];
		    temp[half*7+tid] = d*d* invl1; 
		    }
#endif
		  else if (gtid_l + 2 < numData) {
#if VEC
		    float3 data4=make_float3(data_l[j], data_l[THREADS2+j], data_l[THREADS2*2+j]);
		    float3 temp4=make_float3(data4.x-meansl, data4.y-meansl, data4.z-meansl);
		    float3 temp24 = temp4*temp4;
		    temp24 = temp24 * invl;
		    temp[tid] = temp24.x;
		    temp[half+tid] = temp24.y;
		    temp[THREADS2+tid] = temp24.z;
		  ////////////////
		    temp4=make_float3(data4.x-meansl1, data4.y-meansl1, data4.z-meansl1);
		    temp24 = temp4*temp4* invl1;
		    temp[tid+half*4] = temp24.x;
		    temp[half*5+tid] = temp24.y;
		    temp[THREADS2*3+tid] = temp24.z;
#else
		    d1 = data_l[half*2+j];
		    d = d1 - meansl; //[i_cl*DIM + tid + j];
		    temp[half+tid] = d*d* invl;
		    d = d1 - meansl1; //[i_cl*DIM + tid + j];
		    temp[half*5+tid] = d*d* invl1;

		    d1 = data_l[THREADS2*2+j];
		    d = d1 - meansl;
		    temp[half*2+tid] = d*d* invl; 
		    d = d1 - meansl1;
		    temp[half*6+tid] = d*d* invl1;
#endif
		  }
		  else if (gtid_l + 1 < numData) {
#if VEC
		    
		    float2 data4=make_float2(data_l[j], data_l[THREADS2+j]);
		    float2 temp4=make_float2(data4.x-meansl, data4.y-meansl);
		    float2 temp24 = temp4*temp4;
		    temp24 = temp24 * invl;
		    temp[tid] = temp24.x;
		    temp[half+tid] = temp24.y;
		  ////////////////
		    temp4=make_float2(data4.x-meansl1, data4.y-meansl1);
		    temp24 = temp4*temp4* invl1;
		    temp[tid+half*4] = temp24.x;
		    temp[half*5+tid] = temp24.y;
#else
		    d1 = data_l[THREADS2+j];
		    d = d1 - meansl; //[i_cl*DIM + tid + j];
		    temp[half+tid] = d*d* invl;
		    d = d1 - meansl1; //[i_cl*DIM + tid + j];
		    temp[half*5+tid] = d*d* invl1;
#endif
		  }
		  else {
		    d1 = data_l[j];
		    d = d1 - meansl; //[i_cl*DIM + tid + j];
		    temp[tid] = d*d* invl;
		    d = d1 - meansl1; //[i_cl*DIM + tid + j];
		    temp[half*4+tid] = d*d* invl1;
		  }

		  tid += half;
		}
		else {
#if VEC
		  float d;		  
		  if(gtid_l+3<numData) {
		  
		  float4 data4=make_float4(data_l[j], data_l[THREADS2+j], data_l[THREADS2*2+j], data_l[THREADS2*3+j]);
		  //		  float4 invl4 = make_float4(invl, invl, invl, invl);
		  float4 temp4=make_float4(data4.x-meansl, data4.y-meansl, data4.z-meansl, data4.w-meansl);
		  float4 temp24 = temp4*temp4;
		  temp24 = temp24 * invl;
		  tt0 = temp24.x;
		  tt2 = temp24.y;
		  tt4 = temp24.z;
		  tt6 = temp24.w;
		  ////////////////
		  temp4=make_float4(data4.x-meansl1, data4.y-meansl1, data4.z-meansl1, data4.w-meansl1);
		  temp24 = temp4*temp4* invl1;
		  tt1 = temp24.x;
		  tt3 = temp24.y;
		  tt5 = temp24.z;
		  tt7 = temp24.w;
		  } 
#else
		  TYPE d = data_l[j] - meansl; //[i_cl*DIM + tid + j];
		  tt0 = d*d* invl; //(*(invCovariances + i_cl * DIM + tid + j));
		  d = data_l[j] - meansl1;
		  tt1 = d*d*invl1;
		  
		  if(gtid_l + 3 < numData) {

		    d = data_l[half*2 + j ] - meansl; //[i_cl*DIM + tid + j];
		    tt2 = d*d* invl;
		    d = data_l[half*2 + j ] - meansl1; //[i_cl*DIM + tid + j];
		    tt3 = d*d* invl1;

		    d = data_l[half*4 + j ] - meansl;
		    tt4 = d*d* invl; 
		    d = data_l[half*4 + j] - meansl1;
		    tt5 = d*d* invl1;

		    d = data_l[half*6 + j ] - meansl; //[i_cl*DIM + tid + j];
		    tt6 = d*d* invl; 
		    d = data_l[half*6 + j ] - meansl1; //[i_cl*DIM + tid + j];
		    tt7 = d*d* invl1; 
		  }
#endif
		  else if(gtid_l + 2 < numData) {
#if VEC
		 
		    float3 data4=make_float3(data_l[j], data_l[THREADS2+j], data_l[THREADS2*2+j]);
		    float3 temp4=make_float3(data4.x-meansl, data4.y-meansl, data4.z-meansl);
		    float3 temp24 = temp4*temp4;
		    temp24 = temp24 * invl;
		    tt0 = temp24.x;
		    tt2 = temp24.y;
		    tt4 = temp24.z;
		    ////////////////
		    temp4=make_float3(data4.x-meansl1, data4.y-meansl1, data4.z-meansl1);
		    temp24 = temp4*temp4* invl1;
		    tt1 = temp24.x;
		    tt3 = temp24.y;
		    tt5 = temp24.z;
#else
		    d = data_l[half*2 + j ] - meansl; //[i_cl*DIM + tid + j];
		    tt2 += d*d* invl;
		    d = data_l[half*2 + j ] - meansl1; //[i_cl*DIM + tid + j];
		    tt3 += d*d* invl1;

		    d = data_l[half*4 + j ] - meansl;
		    tt4 += d*d* invl; 
		    d = data_l[half*4 + j] - meansl1;
		    tt5 += d*d* invl1;
#endif
		  }
		  else if(gtid_l + 1 < numData) {
#if VEC
		    float2 data4=make_float2(data_l[j], data_l[THREADS2+j]);
		    float2 temp4=make_float2(data4.x-meansl, data4.y-meansl);
		    float2 temp24 = temp4*temp4;
		    temp24 = temp24 * invl;
		    tt0 = temp24.x;
		    tt2 = temp24.y;
		    ////////////////
		    temp4=make_float2(data4.x-meansl1, data4.y-meansl1);
		    temp24 = temp4*temp4* invl1;
		    tt1 = temp24.x;
		    tt3 = temp24.y;
#else
		    d = data_l[half*2 + j ] - meansl; //[i_cl*DIM + tid + j];
		    tt2 += d*d* invl;
		    d = data_l[half*2 + j ] - meansl1; //[i_cl*DIM + tid + j];
		    tt3 += d*d* invl1;
#endif
		  }
		  else {
		    d = data_l[j ] - meansl; //[i_cl*DIM + tid + j];
		    tt2 += d*d* invl;
		    d = data_l[j ] - meansl1; //[i_cl*DIM + tid + j];
		    tt3 += d*d* invl1;
		  }
		}

	  }
	}
	__syncthreads();
#if 1
	if(tid < THREADS2/2) {
	  temp[ tid] += tt0; //temp[128*i + tid + THREADS2/2];
	  temp[half*4 + tid] += tt1; //emp[128*i+512  + tid + THREADS2/2];
	    temp[half + tid] += tt2; //temp[128*i + tid + THREADS2/2];
	    temp[half*5 + tid] += tt3; //emp[128*i+512  + tid + THREADS2/2];

	    temp[half*2 + tid] += tt4; //temp[128*i + tid + THREADS2/2];
	    temp[half*6 + tid] += tt5; //emp[128*i+512  + tid + THREADS2/2];

	    temp[half*3 + tid] += tt6; //temp[128*i + tid + THREADS2/2];
	    temp[half*7 + tid] += tt7; //emp[128*i+512  + tid + THREADS2/2];

	}	
	__syncthreads();
	////////////The first 24 
	  //i_cl
	  if(tid < THREADS2/4) {
	    for(i=0;i< TILE;i++) {
	      temp[half*i + tid] += temp[half*i + tid + THREADS2/4];
	      temp[half*i + tid+half*4] += temp[half*i + tid + THREADS2/4+half*4];
	    }
	  }
	////////////// The first 12
	if(tid<THREADS2/8) {
	    for(i=0;i< TILE;i++) {
	      temp[half*i + tid] += temp[half*i + tid + THREADS2/8];
	      temp[half*i + tid + half*4] += temp[half*i + tid + THREADS2/8 + half*4];
	    }
	}
	///////////////The first 6
	if(tid < 6) {
	  for(i=0;i< TILE;i++) {
	    temp[half*i + tid] += temp[half*i + tid + 6];
	    temp[half*i + tid + half*4] += temp[half*i + tid + 6 + half*4];
	  }	  
	}
	//////////////////////////////The first 3
	if(tid < 3) {
	  for(i=0;i< TILE;i++) {
	    temp[half*i + tid] += temp[half*i + tid + 3];
	    temp[half*i + tid+half*4] += temp[half*i + tid + 3+half*4];
	  }	  
	}
	if(tid >= THREADS2-4) {
	  p = logWeights[i_cl] - halfDimLog2Pi - 0.5 * logCovariances[i_cl]; 
	  p2 = logWeights[i_cl+1] - halfDimLog2Pi - 0.5 * logCovariances[i_cl+1]; 
	}
	__syncthreads();

	/////////////Using THREADS2-1 provides better load balance
	if(tid >= THREADS2 -4) {
	  //Warning: we know it is the last iteration in the tree, so we omit the loop here. Therefore, we can combine this last step with the following computation
	    i = THREADS2 - tid - 1;
	    {
	      if(gtid_l+i<numData) {
		float max = maxPosterior[i];
		//WARNING: using temp1 instead of temp also gave a small improvement
		float temp1 = temp[half*i] + temp[half*i+1] + temp[half*i+2];
		p1 = p  - 0.5 * temp1;
		posteriors[ i_cl + (gtid_l+i) * numClusters] = p1;
		if(p1> max) { max = p1; }
		
		temp1 = temp[half*i+half*4] + temp[half*i+half*4+1] + temp[half*i+half*4+2];
		p1 = p2 - 0.5*temp1; //[i*64+256];
		posteriors[i_cl + 1+(gtid_l+i)*numClusters] = p1;
		if(p1> max) { max = p1; }
		maxPosterior[i] = max;
	      }
	    }
	  }
#endif
      __syncthreads();
      }//i_cl

      //WARNING: it must be initialized, even though being written to
      for(i=0;i<TILE;i++)
	temp[tid+THREADS2 * i] = 0.0;

#if VEC
      float4 mycluster = {0.0, 0.0, 0.0, 0.0};
#else
      float mycluster[TILE] = {0.0, 0.0, 0.0, 0.0};
#endif
      for(i_cl = tid; i_cl < numClusters; i_cl += THREADS2) {
#if VEC
	if(gtid_l+3<numData) {
	  float4 p4 = make_float4(posteriors[i_cl + (gtid_l)*numClusters],posteriors[i_cl + (gtid_l+1)*numClusters],posteriors[i_cl + (gtid_l+2)*numClusters],posteriors[i_cl + (gtid_l+3)*numClusters]);
	  float4 maxP4=make_float4(maxPosterior[0],maxPosterior[1],maxPosterior[2],maxPosterior[3]); 
	  //	  p4 = p4-maxP4;//exp(p4-maxPosterior[i]); 
	  p4 = make_float4(exp(p4.x-maxP4.x),exp(p4.y-maxP4.y),exp(p4.z-maxP4.z),exp(p4.w-maxP4.w));
	  mycluster = mycluster + p4;
	  posteriors[i_cl + (gtid_l)*numClusters] = p4.x;
	  posteriors[i_cl + (gtid_l+1)*numClusters] = p4.y;
	  posteriors[i_cl + (gtid_l+2)*numClusters] = p4.z;
	  posteriors[i_cl + (gtid_l+3)*numClusters] = p4.w;
	}
	else if (gtid_l+2<numData) {
	  float3 p4 = make_float3(posteriors[i_cl + (gtid_l)*numClusters],posteriors[i_cl + (gtid_l+1)*numClusters],posteriors[i_cl + (gtid_l+2)*numClusters]);
	  float3 maxP4=make_float3(maxPosterior[0],maxPosterior[1],maxPosterior[2]); 
	  //	  p4 = p4-maxP4;//exp(p4-maxPosterior[i]); 
	  p4 = make_float3(exp(p4.x-maxP4.x),exp(p4.y-maxP4.y),exp(p4.z-maxP4.z));
	  //	    mycluster[i] += p; 
	  mycluster.x+=p4.x;
	  mycluster.y+=p4.y;
	  mycluster.z+=p4.z;
	  posteriors[i_cl + (gtid_l)*numClusters] = p4.x;
	  posteriors[i_cl + (gtid_l+1)*numClusters] = p4.y;
	  posteriors[i_cl + (gtid_l+2)*numClusters] = p4.z;
	}
	else if(gtid_l+1<numData) {
	  float2 p4 = make_float2(posteriors[i_cl + (gtid_l)*numClusters],posteriors[i_cl + (gtid_l+1)*numClusters]);
	  float2 maxP4=make_float2(maxPosterior[0],maxPosterior[1]); 
	  p4 = make_float2(exp(p4.x-maxP4.x),exp(p4.y-maxP4.y));
	  mycluster.x+=p4.x;
	  mycluster.y+=p4.y;
	  posteriors[i_cl + (gtid_l)*numClusters] = p4.x;
	  posteriors[i_cl + (gtid_l+1)*numClusters] = p4.y;
	}
	else {
	  float p = posteriors[i_cl + (gtid_l)*numClusters];
	  p = exp(p-maxPosterior[0]); 
	  mycluster.x += p; //maxPosterior[i*THREADS2];
	  posteriors[i_cl + (gtid_l)*numClusters] = p;
	}
#else
	for(i=0;i< TILE;i++) {
	  if(gtid_l+i<numData) {
	    float p = posteriors[i_cl + (gtid_l+i)*numClusters];
	    p = exp(p-maxPosterior[i]); //*THREADS2]);
	    mycluster[i] += p; //maxPosterior[i*THREADS2];
       	    posteriors[i_cl + (gtid_l+i)*numClusters] = p;
	  }
	}
#endif
      }
#if VEC
      temp[tid] = mycluster.x;
      temp[tid+THREADS2] = mycluster.y;
      temp[tid+2*THREADS2] = mycluster.z;
      temp[tid+3*THREADS2] = mycluster.w;
#else
      for(i=0;i<TILE;i++)
	temp[tid+THREADS2 * i] = mycluster[i];
#endif
      __syncthreads();

      //reduction
#if 1
      if(tid < THREADS2/2) {
	for(i=0;i< TILE;i++)
	  temp[THREADS2*i + tid] += temp[THREADS2*i + tid + THREADS2/2];
      }
      __syncthreads();
      /// reduction 
      if(tid < THREADS2/4) {
	for(i=0;i< TILE;i++)
	  temp[THREADS2*i + tid] += temp[THREADS2*i + tid + THREADS2/4];
      }
      //      __syncthreads();
      if(tid < THREADS2/8) {
	for(i=0;i< TILE;i++)
	  temp[THREADS2*i + tid] += temp[tid + THREADS2/8 + THREADS2*i];
      }
      //      __syncthreads();
      if(tid < THREADS2/16) {
	for(i=0;i< TILE;i++)
	  temp[THREADS2*i + tid] += temp[tid + THREADS2/16 + THREADS2*i];
      }
      //      __syncthreads();
      if(tid < THREADS2/32) {
	for(i=0;i< TILE;i++)
	  temp[THREADS2*i + tid] += temp[tid + THREADS2/32 + THREADS2*i];
      }
      //      __syncthreads();
      //WARNING: This last step does not give any benefits, but the last step for temp got visible improvement
      if(tid == 0) {	
	for(int j=1;j<THREADS2/32 && j<DIM;j++) {
	  for(i=0;i< TILE;i++)
	    temp[THREADS2*i] += temp[j+THREADS2*i];
	}
      }
#endif            
      __syncthreads();
#if VEC
      if(gtid_l+3<numData) {
	for (i_cl = tid; i_cl < numClusters; i_cl += THREADS2) {
	  float4 temp24 = make_float4(temp[0], temp[THREADS2], temp[THREADS2*2], temp[THREADS2*3]);
	  float4 post4 = make_float4(posteriors[i_cl+(gtid_l)*numClusters],posteriors[i_cl+(gtid_l+1)*numClusters],posteriors[i_cl+(gtid_l+2)*numClusters],posteriors[i_cl+(gtid_l+3)*numClusters]);
	  post4 = post4/temp24;
	  posteriors[i_cl+(gtid_l)*numClusters] = post4.x;
	  posteriors[i_cl+(gtid_l+1)*numClusters] = post4.y;
	  posteriors[i_cl+(gtid_l+2)*numClusters] = post4.z;
	  posteriors[i_cl+(gtid_l+3)*numClusters] = post4.w;
	}
      }
      else {
	for(i=0;i< TILE;i++) {
	for (i_cl = tid; i_cl < numClusters; i_cl += THREADS2) {
	    if(gtid_l+i<numData) {
	      posteriors[i_cl + (gtid_l+i)*numClusters] /= temp[THREADS2*i];
	    }	  
	}
	}//i
      }
#else
      //    if(tid == 0) { //size) {
      for(i=0;i< TILE;i++) {
	for (i_cl = tid; i_cl < numClusters; i_cl += THREADS2) {
	  //	  if(i_cl + tid < numClusters) {
	    if(gtid_l+i<numData) {
	      posteriors[i_cl + (gtid_l+i)*numClusters] /= temp[THREADS2*i];
	    }	  
	}
      }//i
#endif       
    }//gtid_l

}

__global__ void gmm_3(float*enc, float* data, float* means, float* sqrtInvSigma, float* posteriors, const int dimension, const int numClusters, const int numData, float* priors) {
  int i_cl;
  int gtid = blockIdx.x; // * THREADS2 + threadIdx.x; //get_group_id(0);
  int tid = threadIdx.x;
  int total = numClusters*dimension*2;

  //WARNING: should fix it with the number of clusters
  __shared__ float pl[256*2];
  __shared__ float data_l[THREADS3*2];
  //  __shared__ float data_l2[128];
  float p1,p2; 
  
  for(i_cl = 0; i_cl < numClusters; i_cl ++) {
    
    float* uk_g = enc+i_cl*dimension + gtid * total;
    float* vk_g = enc+i_cl*dimension + numClusters*dimension + gtid * total;
    
    //WARNING: should use "tid" instead of "0", otherwise the checking would be the same for all threads
    for(int dim = tid; dim < dimension; dim+=THREADS3) {
    	*(uk_g + dim ) = 0;
	*(vk_g + dim ) = 0;
    }
  }

  for(int gtid_l  = gtid*ITEMS3; gtid_l<(gtid+1)*ITEMS3 && gtid_l<numData; gtid_l+=ITEMS3/THREADS3) {
    __syncthreads();    
    
    for(int j=tid;j<numClusters;j+=THREADS3) {
      //      if(j<numClusters) 
      {
	//	for(int i=0;i<ITEMS3/THREADS3;i++) {

	pl[j] = posteriors[j+ (gtid_l)*numClusters];
	if(gtid_l + 1<numData) {
	  //WARNING: should use "numClusters" instead of 1024
	  pl[j+numClusters] = posteriors[j+ (gtid_l+1)*numClusters];
	}	
	else {
	  pl[j+numClusters] = 0.0;
	}
      }
    }

    for(int j=tid;j < dimension; j+= THREADS3) 
	for(int i=0;i<ITEMS3/THREADS3;i++) {
	  if(gtid_l + i < numData)  
	    data_l[THREADS3*i+j] = data[(gtid_l+i) * dimension + j];
	}


    __syncthreads();    
    //Process 2 data every time
    p2 = 0.0;
    float p11=0.0, p22=0.0, p12=0.0, p13=0.0, p21=0.0, p23=0.0;
    for(i_cl = 0; i_cl < numClusters; i_cl +=4) {

      p1 = pl[i_cl];
      p11 = pl[i_cl+1];
      p12= pl[i_cl + 2];
      p13 = pl[i_cl + 3];
      
      p2 =pl[i_cl + numClusters]; //posteriors[i_cl + (gtid_l+1)*numClusters];
      p21 = pl[i_cl + 1 + numClusters];
      p22 = pl[i_cl + 2 + numClusters];
      p23 = pl[i_cl + 3 + numClusters];

      //
      //      if (p1<1e-6 && p2 < 1e-6 && p11<1e-6 && p22<1e-6 && p12<1e-6 && p13<1e-6 && p21<1e-6 && p23<1e-6) 
      bool t1 = p1<1e-6,t2 = p2<1e-6,t3 = p11<1e-6,t4 = p13<1e-6,t5 = p12<1e-6,t6 = p21<1e-6,t7 = p22<1e-6, t8 =p23<1e-6;
      
      bool temp1 = t1&t2&t3&t4&t5&t6&t7&t8;
      if(temp1 > 0)
	continue;
      
      float* uk_g = enc+i_cl*dimension + gtid * total;
      float* vk_g = enc+i_cl*dimension + numClusters*dimension + gtid * total;
      
      for(int dim = tid; dim < dimension; dim+=THREADS3) {

	float diff;
      	float meansl = means[i_cl*dimension + dim];
	float meansl1 = means[(i_cl+1)*dimension + dim];

#if 1
	  if(priors[i_cl]>=1e-6)
	  {
	  if( p1 >= 1e-6) {
	    diff= data_l[dim ] - meansl; //means[i_cl*dimension + dim+tid];
	    //diff= data[(gtid_l)*dimension + dim + tid] - means[i_cl*dimension + dim+tid];
	    diff *= sqrtInvSigma[i_cl*dimension + dim ];
	    *(uk_g + dim) += p1*diff;
	    *(vk_g + dim) += p1*(diff*diff-1);
	  }

	  if( p2 >= 1e-6) {
	    diff= data_l[THREADS3+dim] - meansl; //means[i_cl*dimension + dim+tid];
	    //	  diff= data[(gtid_l+1)*dimension + dim + tid] - means[i_cl*dimension + dim+tid];
	    diff *= sqrtInvSigma[i_cl*dimension + dim ];
	    *(uk_g + dim ) += p2*diff;
	    *(vk_g + dim ) += p2*(diff*diff-1);
	  }
	  }
	  if(priors[i_cl+1]>=1e-6) 
	  {
	  if(p11>= 1e-6) {
	    diff= data_l[dim] - meansl1; 
	    diff *= sqrtInvSigma[(i_cl+1)*dimension + dim];
	    *(uk_g + dimension + dim) += p11*diff;
	    *(vk_g + dimension + dim) += p11*(diff*diff-1);
	  }
	  if( p21>= 1e-6) {
	    diff= data_l[THREADS3+dim] - meansl1; 
	    diff *= sqrtInvSigma[(i_cl+1)*dimension + dim];
	    *(uk_g + dimension + dim) += p21*diff;
	    *(vk_g + dimension + dim) += p21*(diff*diff-1);
	  }
	  }

	  if(priors[i_cl+2]>=1e-6) 
	  {
	  if(p12>= 1e-6) {
	    diff= data_l[dim] - means[(i_cl+2)*dimension + dim ]; 
	    diff *= sqrtInvSigma[(i_cl+2)*dimension + dim];
	    *(uk_g + dimension*2 + dim ) += p12*diff;
	    *(vk_g + dimension*2 + dim ) += p12*(diff*diff-1);
	  }
	  if( p22>= 1e-6) {
	    diff= data_l[THREADS3+dim] - means[(i_cl+2)*dimension + dim ]; 
	    diff *= sqrtInvSigma[(i_cl+2)*dimension + dim ];
	    *(uk_g + dimension*2 + dim) += p22*diff;
	    *(vk_g + dimension*2 + dim) += p22*(diff*diff-1);
	  }
	  }
	  
	  if(priors[i_cl+3]>=1e-6) 
	  {
	  if(p13>= 1e-6) {
	    diff= data_l[dim] - means[(i_cl+3)*dimension + dim]; 
	    diff *= sqrtInvSigma[(i_cl+3)*dimension + dim];
	    *(uk_g + dimension*3 + dim ) += p13*diff;
	    *(vk_g + dimension*3 + dim ) += p13*(diff*diff-1);
	  }
	  if( p23>= 1e-6) {
	    diff= data_l[THREADS3+dim] - means[(i_cl+3)*dimension + dim]; 
	    diff *= sqrtInvSigma[(i_cl+3)*dimension + dim];
	    *(uk_g + dimension*3 + dim ) += p23*diff;
	    *(vk_g + dimension*3 + dim ) += p23*(diff*diff-1);
	  }
	  }
#endif
      }
    }
  }
  //////////////global synchronization

}

__global__ void gmm_4(float* enc_d, int dimension, int numClusters, int blocks) {
  int tid = threadIdx.x;
  int gid = blockIdx.x;

  int total = numClusters * dimension;
  if(gid >= numClusters)
    return;

  for (int i= 1;i<blocks;i++) {
    //WARNING: should check dimension
    if(tid<dimension) {
      enc_d[gid * dimension + tid] += enc_d[gid * dimension + tid + i*total*2];
      enc_d[gid * dimension + tid + total] += enc_d[gid * dimension + tid + i*total*2 + total];
    }
  }
}

#define THREADS_P 128
__global__ void gmm_prefix(float* enc, float* priors, int numData, int numClusters, int dimension, int blocks, float* sum) {
  int tid = threadIdx.x;
  int gid = blockIdx.x;

  __shared__ float suml[THREADS_P];

  suml[tid] = 0.0;

  for(int i_cl = gid; i_cl < numClusters;i_cl +=blocks) 
  {
    float* uk_g = enc+i_cl*dimension;
    float* vk_g = enc+i_cl*dimension + numClusters*dimension;
    //WARNING: it must be here also!
    if(priors[i_cl]<1e-6) 
      continue;

    for(int dim =tid; dim<dimension;dim += THREADS_P) {
      //      *(uk_g+dim) = *(uk_g+dim) * (1/(numData * sqrt(priors[i_cl])));
      //*(vk_g+dim) *= 1/(numData * sqrt(2*priors[i_cl]));
    
      float ukg = *(uk_g+dim) * (1/(numData * sqrt(priors[i_cl])));
      if(ukg>0)
	*(uk_g+dim) = sqrt(ukg);
      else 
	*(uk_g+dim) = -sqrt(-ukg);
      float vkg = *(vk_g+dim) * (1/(numData * sqrt(2*priors[i_cl])));
      if(vkg>0) 
	*(vk_g+dim) = sqrt(vkg);
      else
	*(vk_g+dim) = -sqrt(-vkg);

      suml[tid] += (*(uk_g+dim)) *(*(uk_g+dim)) + (*(vk_g+dim)) *(*(vk_g+dim));
    }	  
  }

  __syncthreads();
  
      if(tid < THREADS_P/2) {
	  suml[tid] += suml[tid + THREADS_P/2];
      }
      __syncthreads();
      /// reduction 
      if(tid < THREADS_P/4) {
	  suml[tid] += suml[tid + THREADS_P/4];
      }
      if(tid < THREADS_P/8) {
	  suml[tid] += suml[tid + THREADS_P/8];
      }
      if(tid < THREADS_P/16) {
	suml[tid] += suml[tid + THREADS_P/16];
      }
      //      __syncthreads();
      if(tid < THREADS_P/32) {
	suml[tid] += suml[tid + THREADS_P/32];
      }
      //      __syncthreads();
      //WARNING: This last step does not give any benefits, but the last step for temp got visible improvement
      if(tid < THREADS_P/64) {
	suml[tid] += suml[tid + THREADS_P/64];
      }
      if(tid == 0) {	
	suml[tid] += suml[tid + 1];
	sum[gid] = suml[tid];
      }
}
__global__ void prefix_norm(float* enc, int numData, int numClusters, int dimension, int blocks, float n) {
  int tid = threadIdx.x; 
  int gid = blockIdx.x; 

  for(int i_cl = gid; i_cl < numClusters;i_cl +=blocks) 
  {
    if(i_cl >= numClusters)
      break;

    float* uk_g = enc+i_cl*dimension;
    float* vk_g = enc+i_cl*dimension + numClusters*dimension;

    for(int dim =tid; dim<dimension;dim += THREADS_P) {
      *(uk_g+dim) = *(uk_g+dim) * (1/n);
      *(vk_g+dim) *= 1/(n);
    }
  }
}

#define MAX_FILT 512
__global__ void sift(float* dstp, int dst_stride, float* srcp, int src_width, int src_height, int src_stride, float* filtp, int filt_ker) {
  int tid = threadIdx.x; 
  int gid = blockIdx.x; 

  //  int x = 0;
  float* dst, *src;
  int y; //y is tid, x is gid

  float* filt;

  //dst_stride == src_height == dheight;
  //src_stride == src_width

  float* my_dst;

  __shared__ float temp_src[MAX_FILT];
  float* filti;

  int filt_begin = -filt_ker;
  int filt_end = filt_ker;
  // int filt_size = filt_end - filt_begin;
  int image_size = src_height*src_width;

  for(int bint = 0; bint<8;bint++) 
  {
    src = srcp + bint*image_size;
    int count = 0;
    /*
      __syncthreads();
      for(y=tid;y<MAX_FILT;y+=THREADS_S) {
      temp_src[y] = 0.0;
      }
    */
    int set = src_height/THREADS_S;
      if(src_height%THREADS_S)
	set ++;
      set *= THREADS_S;

    for(y=tid;y<set; y+= THREADS_S) {

      __syncthreads();

      if(y<THREADS_S) {
	if(tid>=filt_end && y-filt_end<src_height) {
	  temp_src[tid-filt_end] = src[gid + (y-filt_end) *src_width] ; // + y*src_width];
	}
	if(tid < filt_end-filt_begin + 1 && y-filt_end+THREADS_S < src_height) {
	  temp_src[tid - filt_end +THREADS_S] = src[gid + (y-filt_end + THREADS_S ) * src_width];
	}
      }
      else  {
	//In the second and following iterations
	if( y-filt_end<src_height) {
	  //temp_src[0] is src[gid + 124*src_width], load till src[gid + 251*src_width]
	  temp_src[tid] = src[gid + (y-filt_end) *src_width] ; 
	}
	if(tid < filt_end-filt_begin + 1 && y-filt_end+THREADS_S < src_height) {
	  //temp_src[128] is src[gid + 252*src_width]
	  temp_src[tid+THREADS_S] = src[gid + (y-filt_end +THREADS_S) * src_width];
	}
      }

      __syncthreads();
      if(y>=src_height)
	break;

      int offset = gid*src_height + y;

      for(int biny = 0; biny < 4; biny++) {

	filt = filtp + biny*10;
	dst = dstp + image_size*(biny + 4*bint);
	filt += filt_end - filt_begin;

	float c, v = 0;
	float acc = 0;
	int stop = filt_end - y;

	int srci = 0;

	filti = filt;
	if (stop > 0) {
	  v = temp_src[0]; //(src + gid);
	  for(filti=filt; filti > filt - stop; filti--) {
	    //      for(int i=0;i<stop;i++) {
	    c = *filti;
	    acc += v * c;
	    //	progress++;
	  }
	}
	else {
	  srci -= stop; //*src_stride;
	  //In second iter, srci = 124 ~ 251
	}
    
	//The second while
	stop = filt_end - VL_MAX(filt_begin, y - src_height + 1) + 1;
	for(; filti > filt- stop; filti--) {
	  //    for(int i=0;i< stop-progress;i++) {
	  if(y<THREADS_S)
	    //	    v = src[srci*src_stride + gid];
       	    v = temp_src[srci]; //src[srci*src_stride + gid]; //temp_src[srci]; ////*srci;
	  else 
	    v = temp_src[srci +filt_end - THREADS_S*count]; //src[srci*src_stride + gid];
	  //In second iteration, src[124*src_stride + gid] loop to ...
	  c = *filti;
	  acc += v * c;
	  srci++;
	}

	//The third while
	stop = filt_end - filt_begin + 1;

	for(; filti > filt - stop; filti--) {
	  //    for(int i=0;i<stop - progress; i++) {
	  c = *filti;
	  acc += v * c;
	}
	//      for(int binx = 0 ;binx<4;binx++) {
	my_dst = dst + offset;
	*my_dst = acc;
      }//binx
      count++;

    } //y

  }
  
}

__global__ void sift2(float* dstp, int dst_stride, float* srcp, int src_width, int src_height, int src_stride, float * filtp, int filt_ker) {
  //filtp is in constant memory
  int tid = threadIdx.x;
  int gid = blockIdx.x;

  float* dst, *src = srcp;
  gid = gid*5;
  
  int y; //y is tid, x is gid

  float* filt;

  //dst_stride == src_height == dheight;
  //src_stride == src_width

  float* my_dst;

  __shared__ float temp_src[MAX_FILT];
  float* filti;

  int width_range = (src_height-16)/5 + 1;
  int height_range = (src_width-16)/5 + 1;
  //int image_size = src_height*(src_width/5+1);

  for(int bint = 0; bint< 8; bint++) 
  {
    for(int biny=0;biny<4;biny++) {
      src = srcp + src_height*src_width * (biny+4*bint);

      __syncthreads();

      int filt_begin = -filt_ker;
      int filt_end = filt_ker;
      //      int filt_size = filt_end - filt_begin;

      //  __shared__ float* temp_filt = filter + filt_size;
      int count = 0;

      int set = src_height/THREADS_S;
      if(src_height%THREADS_S)
	set ++;
      set *= THREADS_S;

      for(y=tid;y<set; y+= THREADS_S) {

	__syncthreads();

	if(y<THREADS_S) {
	  if(tid>=filt_end && y-filt_end<src_height) {
	    temp_src[tid-filt_end] = src[gid + (y-filt_end) *src_width] ; // + y*src_width];
	  }
	  if(tid < filt_end-filt_begin + 1 && y-filt_end+THREADS_S < src_height) {
	    temp_src[tid - filt_end +THREADS_S] = src[gid + (y-filt_end + THREADS_S ) * src_width];
	  }
	}
	else  {
	  //In the second and following iterations
	  if( y-filt_end<src_height) {
	    //temp_src[0] is src[gid + 124*src_width], load till src[gid + 251*src_width]
	    temp_src[tid] = src[gid + (y-filt_end) *src_width] ; // + y*src_width];
	  }
	  if(tid < filt_end-filt_begin + 1 && y-filt_end+THREADS_S < src_height) {
	    //temp_src[128] is src[gid + 252*src_width]
	    temp_src[tid+THREADS_S] = src[gid + (y-filt_end +THREADS_S) * src_width];
	  }
	}

	__syncthreads();

	if(y>=src_height)
	  break;
      
	int offset = gid/5*src_height + y;		
	for(int binx = 0; binx<4;binx++) {
      
	  filt = filtp + binx*10;

	  filt += filt_end - filt_begin;

	  float c, v = 0;
	  float acc = 0;
	  int stop = filt_end - y;

	  int srci = 0;

	  filti = filt;

	  if (stop > 0) {
	    v = temp_src[0]; //(src + gid);
	    for(; filti > filt - stop; filti--) {
	      c = *filti;
	      acc += v * c;
	    }
	  }
	  else {
	    srci -= stop; //*src_stride;
	    //In second iter, srci = 124 ~ 251
	  }

	  //The second while
	  stop = filt_end - VL_MAX(filt_begin, y - src_height + 1) + 1;
	  for(; filti > filt- stop; filti--) {
	    //    for(int i=0;i< stop-progress;i++) {
	    if(y<THREADS_S)
	      v = temp_src[srci]; //src[srci*src_stride + gid]; //temp_src[srci]; ////*srci;
	    else //if(count<src_height/THREADS_S || y < src_height - filt_end + filt_begin )
	      v = temp_src[srci +filt_end - THREADS_S*count]; //src[srci*src_stride + gid];
	    //In second iteration, src[124*src_stride + gid] loop to ...
	    c = *filti;
	    acc += v * c;
	    srci++;
	    //      progress ++;
	  }
	  //The third while
	  stop = filt_end - filt_begin + 1;

	  for(; filti > filt - stop; filti--) {
	    c = *filti;
	    acc += v * c;
	  }
	  
	  if(y/5>=width_range+binx || y/5<binx || y%5!=0 || gid/5<biny || gid/5>=height_range+biny) {
	    continue;
	  }
	  else {
	    offset = y/5-binx + (gid/5-biny)*width_range;
	    dst = dstp + 128*offset +(biny*32+binx*8+bint); 
	    my_dst = dst; 
	  }
	  *my_dst = acc;
	}      
	count++;	
      }//y
    } //biny
  }
}

inline __device__ float gpu_vl_fast_resqrt_f (float x)
{
  /* 32-bit version */
  union {
    float x ;
    int  i ;
  } u ;

  float xhalf = (float) 0.5 * x ;

  /* convert floating point value in RAW integer */
  u.x = x ;

  /* gives initial guess y0 */
  u.i = 0x5f3759df - (u.i >> 1);
  /*u.i = 0xdf59375f - (u.i>>1);*/

  /* two Newton steps */
  u.x = u.x * ( (float) 1.5  - xhalf*u.x*u.x) ;
  u.x = u.x * ( (float) 1.5  - xhalf*u.x*u.x) ;
  return u.x ;
}

inline __device__ float fast_sqrt(float x) {
  return ((x) < 1e-8) ? 0 : (x) * gpu_vl_fast_resqrt_f (x) ;
}

//#pragma OPENCL EXTENSION cl_khr_fp64: enabled
#define VL_EPSILON_F 1.19209290E-07F
#define TILE1 1

__global__ void sift_normalize(float* data, int total) {
  //  __global float* mydata;
  int  dimension = 128;

  int tid = threadIdx.x;
  int gid = blockIdx.x;
  //  float norm = 0.0;
  __shared__ float data_l[THREADS_S];
  __shared__ float sum_l[THREADS_S];

  __shared__ float norm_l[1];

  for(gid = blockIdx.x;gid < total;gid += gridDim.x) {

  for(int j = tid; j<dimension; j += THREADS_S) 
    if(j < dimension) {
      data_l[j] = data[gid * dimension + j];
      sum_l[j] = data_l[j] * data_l[j];
    }
  __syncthreads();

  if(tid < THREADS_S/2) {
      sum_l[ tid] += sum_l[tid + THREADS_S/2];
  }
  __syncthreads();
  /// reduction 
  if(tid < THREADS_S/4) {
      sum_l[ tid] += sum_l[ tid + THREADS_S/4];
  }
  //  __syncthreads();
  if(tid < THREADS_S/8) {
      sum_l[ tid] += sum_l[tid + THREADS_S/8 ];
  }
  if(tid < THREADS_S/16) {
      sum_l[tid] += sum_l[tid + THREADS_S/16];
  }
  if(tid < THREADS_S/32) {
      sum_l[ tid] += sum_l[tid + THREADS_S/32];
  }
  //WARNING: This last step does not give any benefits, but the last step for temp got visible improvement
  if(tid < THREADS_S/64) {
      sum_l[tid] += sum_l[tid + THREADS_S/64];
  }
  if(tid == 0) {	
    for(int j=1;j<THREADS_S/64 && j<dimension;j++) {
	sum_l[0] += sum_l[j];
    }
    //float norm = sum_l[0];
    float norm = fast_sqrt(sum_l[0]); 
    norm += VL_EPSILON_F;
    norm_l[0] = norm;
  }
  __syncthreads();

  data_l[tid] /= norm_l[0];
#if 1
  if(data_l[tid]>0.2F)
    data_l[tid] = 0.2F;

  sum_l[tid] = data_l[tid] * data_l[tid];
  ///////////////////////////
  //////////Second normalization
  /////////////////////////
  __syncthreads();

  if(tid < THREADS_S/2) {
      sum_l[tid] += sum_l[ tid + THREADS_S/2];
  }
  __syncthreads();
  /// reduction 
  if(tid < THREADS_S/4) {
      sum_l[ tid] += sum_l[ tid + THREADS_S/4];
  }
  //  __syncthreads();
  if(tid < THREADS_S/8) {
      sum_l[tid] += sum_l[tid + THREADS_S/8];
  }
  if(tid < THREADS_S/16) {
      sum_l[tid] += sum_l[tid + THREADS_S/16];
  }
  if(tid < THREADS_S/32) {
      sum_l[ tid] += sum_l[tid + THREADS_S/32];
  }
  //WARNING: This last step does not give any benefits, but the last step for temp got visible improvement
  if(tid < THREADS_S/64) {
      sum_l[tid] += sum_l[tid + THREADS_S/64];
  }
  if(tid == 0) {	
    for(int j=1;j<THREADS_S/64 && j<dimension;j++) {
	sum_l[0] += sum_l[j];
    }
    float norm = fast_sqrt(sum_l[0]) + VL_EPSILON_F;
    //  norm = sqrt(norm) + VL_EPSILON_F;
    norm_l[0] = norm;
  }
  __syncthreads();

  //  data_l[tid] /= norm_l[0];
#endif
  float tmp = min(data_l[tid]/norm_l[0] * 512, 255.0);  
  data_l[ tid ] = tmp;

  //////////////////The next normalization (the 3rd)
  sum_l[tid] = data_l[tid] * data_l[tid];
  __syncthreads();

  if(tid < THREADS_S/2) {
      sum_l[ tid] += sum_l[tid + THREADS_S/2];
  }
  __syncthreads();
  /// reduction 
  if(tid < THREADS_S/4) {
      sum_l[ tid] += sum_l[ tid + THREADS_S/4];
  }
  //  __syncthreads();
  if(tid < THREADS_S/8) {
      sum_l[ tid] += sum_l[tid + THREADS_S/8 ];
  }

  if(tid < THREADS_S/16) {
      sum_l[ tid] += sum_l[tid + THREADS_S/16 ];
  }

  if(tid < THREADS_S/32) {
      sum_l[ tid] += sum_l[tid + THREADS_S/32];
  }

  //WARNING: This last step does not give any benefits, but the last step for temp got visible improvement
  if(tid < THREADS_S/64) {
      sum_l[ tid] += sum_l[tid + THREADS_S/64];
  }

  if(tid == 0) {	
    for(int j=1;j<THREADS_S/64 && j<dimension;j++) {
	sum_l[0] += sum_l[j];
    }
    float norm = max(1e-5, sqrt(sum_l[0]));
    //  norm = sqrt(norm) + VL_EPSILON_F;
    norm_l[0] = norm;
  }
  __syncthreads();
  tmp = data_l[tid]/norm_l[0];
  data[gid*dimension + tid ] = tmp;
  }
  //  data[gid*dimension + tid ] = data_l[tid]/norm_l[0];  
}

#define VL_PI 3.141592653589793
inline __device__ float gpu_vl_mod_2pi_f (float x)
{
  while (x > (float)(2 * VL_PI)) x -= (float) (2 * VL_PI) ;
  while (x < 0.0F) x += (float) (2 * VL_PI);
  return x ;
}
inline __device__ float gpu_vl_fast_atan2_f (float y, float x)
{
  float angle, r ;
  float const c3 = 0.1821F ;
  float const c1 = 0.9675F ;
  //  float abs_y    = vl_abs_f (y) + VL_EPSILON_F ;
  float abs_y = fabs(y) + VL_EPSILON_F;

  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y) ;
    angle = (float) (VL_PI / 4) ;
  } else {
    r = (x + abs_y) / (abs_y - x) ;
    angle = (float) (3 * VL_PI / 4) ;
  }
  angle += (c3*r*r - c1) * r ;
  return (y < 0) ? - angle : angle ;
}

inline __device__ long int gpu_vl_floor_f(float x) {
    long int xi = (long int) x ;
    if (x >= 0 || (float) xi == x) return xi ;
    else return xi - 1 ;
}

#define at(x,y) (image[(y)*width + (x)])
#define THREADS_G 256

__global__ void init_grads(float* image, float* grads, int width, int height) {
  int gid = blockIdx.x;
  int tid = threadIdx.x;

  float gx, gy ;
  float angle, mod, nt, rbint ;
  int bint ;
  int y = gid;

  for(int x=tid;x<width;x+= THREADS_G) {

    if (y == 0) {
      gy = at(x,y+1) - at(x,y) ;
    } else if (y == height - 1) {
      gy = at(x,y) - at(x,y-1) ;
    } else {
      gy = 0.5F * (at(x,y+1) - at(x,y-1)) ;
    }

    
    if (x == 0) {
      gx = at(x+1,y) - at(x,y) ;
    } else if (x == width - 1) {
      gx = at(x,y) - at(x-1,y) ;
    } else {
      gx = 0.5F * (at(x+1,y) - at(x-1,y)) ;
    }

      
    angle = gpu_vl_fast_atan2_f (gy,gx) ;

    mod = fast_sqrt (gx*gx + gy*gy) ;

    
    nt = gpu_vl_mod_2pi_f (angle) * (8 / (2*VL_PI)) ;

    bint = (int) gpu_vl_floor_f (nt) ;
    rbint = nt - bint ;
      
    int image_size = width * height;
    //    bint = 0;
  
    grads [(bint % 8)*image_size + x + y * width] =  (1 - rbint) * mod ;
    grads [((bint + 1) % 8 )*image_size + x + y * width] = (    rbint) * mod ;
  
    //    grads[tid] = tid;
  }
}
#define THREADX 128
#define THREADY 128
#define INC(x,l) ((x+1) >= (l) ? (x):((x)+1))
__global__ void resize(float* dst, float* src, int dst_offset, int src_offset, int dst_step, int src_step, int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify) {
  int dx = threadIdx.x + blockIdx.x * THREADX;
  int dy = threadIdx.y + blockIdx.y * THREADY;
  float sx = (( dx+0.5f ) * ifx - 0.5f);
  float sy = (( dy+0.5f ) * ify - 0.5f);
  int x = floor(sx), y = floor(sy);
  float diffu = sx-x, diffv = sy-y;
  
  if (x<0)  {
    x=0, diffu = 0;
  }
  if ( x>=src_cols ) x=src_cols-1,diffu=0;
  if ( y<0 ) y=0,diffv=0;
  if (y>=src_rows ) y=src_rows-1,diffv=0;
  //TODO, finish it

  int y_ = INC(y, src_rows);
  int x_ = INC(x,src_cols);
  float u1 = 1.f-diffu;
  float v1 = 1.f-diffv;
  /*
    int4 srcpos;
    srcpos.x = mad24(y, src_step, x+src_offset);
    srcpos.y = mad24(y, src_step, x_+src_offset);
    srcpos.z = mad24(y_, src_step, x+src_offset);
    srcpos.w = mad24(y_, src_step, x_+src_offset);
  */
  int4 srcpos;
  srcpos.x = y*src_step + x+src_offset;
  srcpos.y = y*src_step + x_+src_offset;
  srcpos.z = y_*src_step + x+src_offset;
  srcpos.w = y_*src_step + x_+src_offset;

    float data0 = src[srcpos.x];
    float data1 = src[srcpos.y];
    float data2 = src[srcpos.z];
    float data3 = src[srcpos.w];
    float val1 = u1 *  data0 +
                diffu  *  data1 ;
    float val2 = u1 *  data2 +
                diffu *  data3;
    float val = v1 * val1 + diffv * val2;
    int dstpos = dy*dst_step + dx + dst_offset; //mad24(dy, dst_step, dx+dst_offset);
    if(dx>=0 && dx<dst_cols && dy>=0 && dy<dst_rows)
         dst[dstpos] = val;
}

/*
__kernel void resizeLN_C1_D5(__global float * dst, __global float * src,
                     int dst_offset, int src_offset,int dst_step, int src_step,
                     int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    float sx = ((dx+0.5f) * ifx - 0.5f), sy = ((dy+0.5f) * ify - 0.5f);
    int x = floor(sx), y = floor(sy);
    float u = sx - x, v = sy - y;

    if ( x<0 ) x=0,u=0;
    if ( x>=src_cols ) x=src_cols-1,u=0;
    if ( y<0 ) y=0,v=0;
    if (y>=src_rows ) y=src_rows-1,v=0;

    int y_ = INC(y,src_rows);
    int x_ = INC(x,src_cols);
    float u1 = 1.f-u;
    float v1 = 1.f-v;
    int4 srcpos;
    srcpos.x = mad24(y, src_step, x+src_offset);
    srcpos.y = mad24(y, src_step, x_+src_offset);
    srcpos.z = mad24(y_, src_step, x+src_offset);
    srcpos.w = mad24(y_, src_step, x_+src_offset);
    float data0 = src[srcpos.x];
    float data1 = src[srcpos.y];
    float data2 = src[srcpos.z];
    float data3 = src[srcpos.w];
    float val1 = u1 *  data0 +
                u  *  data1 ;
    float val2 = u1 *  data2 +
                u *  data3;
    float val = v1 * val1 + v * val2;
    int dstpos = mad24(dy, dst_step, dx+dst_offset);
    if(dx>=0 && dx<dst_cols && dy>=0 && dy<dst_rows)
         dst[dstpos] = val;
}
*/
__global__ void kernel1(float* tmpsum, float* data, float* projectionCenter, int numData) {
  int tid = threadIdx.x;
  int gid = blockIdx.x;
  __shared__ float tempcenter[128];
  tempcenter[tid] = projectionCenter[tid];
  __syncthreads();
  for(int i=gid;i<numData;i+=gridDim.x) { //blockDim.x) {
    tmpsum[i*128 + tid] = data[i*128+tid] - tempcenter[tid];
  }
}

//WARNING: in caltech256, a picture may be wider than 500, thus, MAXHEIGHT need to be set higher.
#define MAXHEIGHT 2900
#define MAXFILTER 32
#define THREAD_F 256
__global__ void imconvcoltri(float* dst, int imageHeight, float* src, int imageWidth,int filterSize) {
  int gid = blockIdx.x;
  int tid = threadIdx.x;
  float scale = (float)(1.0/((double)filterSize*(double)filterSize));

  //  __shared__ float bufs[MAXHEIGHT + MAXFILTER];
  __shared__ float imagebuffer[MAXHEIGHT];
  __shared__ float bufs2[MAXHEIGHT + MAXFILTER];

  for(int bint=0;bint<8;bint++) {
    float* src_d = src + imageHeight*imageWidth*bint;

    float* buffer2 = bufs2+filterSize;
    //gid is x
    float* imagei;
    //imageStride = imageWidth
    imagei = src_d + gid; // + imageWidth*(imageHeight -1);
    //  if(tid<imageHeight) {
    for(int i=tid;i<imageHeight;i+=THREAD_F) {
      imagebuffer[i] = *(imagei+imageWidth*i);
    }
    __syncthreads();

    //TODO: now assume the number of threads is more than the height
    for(int i=tid;i<imageHeight+filterSize;i+=THREAD_F) {
      if(i < imageHeight)
      {
	//bufs2[tid] = bufs[tid] - bufs[tid + filterSize];
	bufs2[i] = 0.0;
	if(i>=filterSize ) {
	  for(int j=0;j<filterSize;j++) {
	    bufs2[i] += imagebuffer[i -filterSize + j];
	  }
	}
	else {
	  int j;
	  for(j=0;j<filterSize - i;j++) {
	    bufs2[i] += imagebuffer[0];
	  }
	  for(;j<filterSize;j++) {
	    bufs2[i]+=imagebuffer[j-filterSize + i];
	  }
	}
      }
      else if(i >= imageHeight && i <imageHeight + filterSize) {
      //    bufs2[tid] = bufs[tid] - buffer[imageHeight - 1] *(imageHeight - filterSize - (tid-filterSize) );
	bufs2[i] = 0.0;
	for(int j=0;j<imageHeight+filterSize - i;j++) {
	  bufs2[i] += imagebuffer[i-filterSize + j];
	}
	bufs2[i] -= imagebuffer[imageHeight -1] * (imageHeight -i);
	}
    }
    /* integrate forward the column 
       for (y = - (signed)filterSize + 1 ;
       y < (signed)imageHeight ; ++y) {
       buffer[y] += buffer[y - 1] ;
       }
    */
    __syncthreads();
    /* compute the filter backward 
       {
       vl_size stride = transp ? 1 : destStride ;
       dest += dheight * stride ;
       for (y = step * (dheight - 1) ; y >= 0 ; y -= step) {
       dest -= stride ;
       *dest = scale * (buffer[y] - buffer[y - (signed)filterSize]) ;
       }
       dest += transp ? destStride : 1 ;
       }
    */
    //dheight = imageHeight
    //tid: 0:height-1

    for(int i=tid;i<imageHeight;i+=THREAD_F) {
      float* dest = dst + bint*imageHeight *imageWidth + imageHeight * gid + i; // dheight;
      float tmp = 0.0;
	for(int j=0;j<filterSize;j++) {
	tmp += buffer2[i-j];
	}
      *dest = scale * tmp ; //(buffer[tid] - buffer[tid - filterSize]);
    }
  }
}

__global__ void imconvcoltri2(float* dst, int imageHeight, float* src, int imageWidth,int filterSize, float* w_d) {
  int gid = blockIdx.x;
  int tid = threadIdx.x;
  float scale = (float)(1.0/((double)filterSize*(double)filterSize));

  //  __shared__ float bufs[MAXHEIGHT + MAXFILTER];
  __shared__ float bufs2[MAXHEIGHT + MAXFILTER];
  __shared__ float imagebuffer[MAXHEIGHT];

  //float* buffer = bufs + filterSize;

  //gid is x
  float* imagei;
  //imageStride = imageWidth
  gid *=4;
  for(int bint = 0; bint<8;bint++) 
  {
    float* src_d = src+imageHeight*imageWidth*bint;
    float* buffer2 = bufs2+filterSize;
    __syncthreads();
    imagei = src_d + gid; // + imageWidth*(imageHeight -1);
    for(int i=tid;i<imageHeight;i+=THREAD_F) {
      imagebuffer[i] = *(imagei+imageWidth*i);
    }
    __syncthreads();

    for(int i=tid;i<imageHeight+filterSize;i+=THREAD_F) {
      if(i < imageHeight)
	{
	  bufs2[i] = 0.0;
	  if(i>=filterSize ) {
	    for(int j=0;j<filterSize;j++) {
	      bufs2[i] += imagebuffer[i -filterSize + j];
	    }
	  }
	  else {
	    int j;
	    for(j=0;j<filterSize - i;j++) {
	      bufs2[i] += imagebuffer[0];
	    }
	    for(;j<filterSize;j++) {
	      bufs2[i]+=imagebuffer[j-filterSize + i];
	    }
	  }
	}
      else if(i >= imageHeight && i <imageHeight + filterSize) {
	bufs2[i] = 0.0;
	for(int j=0;j<imageHeight+filterSize - i;j++) {
	  bufs2[i] += imagebuffer[i-filterSize + j];
	}
	bufs2[i] -= imagebuffer[imageHeight -1] * (imageHeight -i);
      }
    }
    __syncthreads();
  

  
  int offset = imageHeight/4;
  if(imageHeight%4>0)
    offset +=1;

  for(int i=tid;i<offset; i+=THREAD_F) { //imageHeight;i+=THREAD_F) {
      //      if(i<offset) 
    {

	int myi = i*4;
	float* dest = dst + bint*imageHeight *imageWidth + imageHeight * gid + myi ;
	//	float* dest = dst + imageHeight * gid + tid; // dheight;
	float tmp = 0.0;
	for(int j=0;j<filterSize;j++) {
	  tmp += buffer2[myi-j];
	}
    /*    int binx = tid/filterSize;
    int biny = gid/filterSize;
    float myw = w_d[biny + binx*filterSize];
    */
	*dest = scale * tmp ; //(buffer[tid] - buffer[tid - filterSize]);
      }
    }
    
  }
}
//16 threads per block: 4X4
//imageHeight*width blocks
__global__ void multWX(float* in, float* out, float* W_d, int filterSize, int height, int width, int whole_height, int whole_width) {
  int gx = blockIdx.x;
  int gy = blockIdx.y;
  int tidx = (threadIdx.x)/32;
  int tidy = ((threadIdx.x)%32)/8;
  int tidt = ((threadIdx.x)%32)%8;

  {
    //write to 16 elements in the output
    int offset = tidy + filterSize * tidx;

    //    int src_offset = gx*step + binsizeX * tidx + (gy*step +binsizeY * tidy)*width;
    int src_offset = gx*4 + 8 * tidx + (gy*4 +8 * tidy)*whole_width + tidt*whole_height*whole_width;
    int dst_offset = (gx*height + gy)*128 + (tidx+tidy*4)*8 + tidt;
    out[dst_offset] = in[src_offset]*W_d[offset]; 
   //src_offset%(whole_height*whole_width); //in[src_offset]; //*W_d[offset];
  }
}

__global__ void get_frame(float deltaCenterX, float deltaCenterY, float* dst, int total, int totalx, int totaly, int step, float scale) { 
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  int indexx = (tid/2)/totaly;
  int indexy = (tid/2)%totaly;
  float tmp;
  if(tid < total) {
    if(tid %2 == 0 ) {
      int index = indexx*step;
      tmp = index + deltaCenterX;
      tmp = ((tmp)/scale) + 1;
    }
    else {
      int index = indexy*step;
      tmp = index + deltaCenterY;
      tmp = ((tmp)/scale) + 1;
    }
    dst[tid] = tmp;
  }
}
#define DST_DIM 80
__global__ void add_frame(float* dest, float* siftframe, float width, float height, float halfwidth, float halfheight, int total) {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  int indexx = tid%2;
  int indexy = tid/2;
  if(tid < total) {
    if(indexx==0) {
      dest[(DST_DIM+2)*indexy + DST_DIM] = (siftframe[indexy*2]-halfwidth)/width;
    }
    else {
      dest[(DST_DIM+2)*indexy + DST_DIM+1] = (siftframe[indexy*2+1]-halfheight)/height;
    }
  }
}
/////////////////////kernels for resize
//#define MAX_WIDTH 2048
//#define MAX_HEIGHT
//WARNING: MAXP is fixed for now
#define MAXP 16
__global__ void resize_h(float* mtxinter1, float* mtx, int height, int width, float scale, float kernelwidth, int antialiasing) {
  //Calculate indicesh and weighth
  //Each block calculates 1 row
  int i_index = blockIdx.x;
  int tid = threadIdx.x;
  //  __shared__ float mtx_s[MAX_WIDTH];
  //Load the data into shared memory
  /*
  for(int j = tid; j<width;j+=blockDim.x) {
    mtx_s[j] = mtx[i_index * width + j];
  }
  __syncthreads();
  */
  
  double mat_u = (double)(i_index+1)/scale+0.5*(1-1/scale);
  //kernel_width is constantly 2, so we fix it.
  int left = floor(mat_u- kernelwidth/2);
  int indicesh[MAXP];
  double weighth[MAXP];
  int P = ceil(kernelwidth) + 2;
  //WARNING: for now, we only calculate the ones that are used (1, 2)
  double sum = 0.0;  
  for(int j=1;j<P-1;j++) {
    indicesh[j-1] = left + j;
    double inter1 = mat_u - indicesh[j-1];

    if(antialiasing && scale<1)
      inter1 = inter1*scale;

    if(inter1 >= -1 && inter1<0) {
      weighth[j-1] = 1+ inter1;
    }
    else if(inter1>=0 && inter1<=1) {
      weighth[j-1] = 1-inter1;
    }
    else {
      weighth[j-1] = 0;
    }

    if(antialiasing && scale<1)
      weighth[j-1] = weighth[j-1]*scale;

    sum += weighth[j-1];
    indicesh[j-1] = min(height, max(1, indicesh[j-1]));
  }
  for(int j=1;j<P-1;j++) 
    if(sum!=0)
      weighth[j-1] = weighth[j-1]/sum;

  //Calculate mtxinter1. Each thread calculate one element in width
  for(int j=tid;j<width;j+=blockDim.x) {
    mtxinter1[i_index*width + j] = 0.0;
    for(int k=1;k<P-1;k++)
      mtxinter1[i_index*width + j] += mtx[(indicesh[k-1] - 1)*width + j] * weighth[k-1];
  }
}
__global__ void resize_w(float* out, float* mtxinter1, int height, int width ,float scale, int new_h, int new_w, float kernelwidth, int antialiasing) {
  //Calculate indicesh and weighth
  //Each block calculates 1 row
  int i_index = blockIdx.x;
  int tid = threadIdx.x;
  //  __shared__ float mtx_s[MAX_HEIGHT];
  //Load the data into shared memory
  /*
  for(int j = tid; j<width;j+=blockDim.x) {
    mtx_s[j] = mtxinter1[i_index * width + j];
  }
  __syncthreads();
  */
  double mat_u = (double)(i_index+1)/scale+0.5*(1-1/scale);
  //kernel_width is constantly 2, so we fix it.
  int left = floor(mat_u- kernelwidth/2);
  int indicesw[MAXP];
  double weightw[MAXP];
  int P = ceil(kernelwidth) + 2;
  //WARNING: for now, we only calculate the ones that are used (1, 2)
  double sum = 0.0;
  for(int j=1;j<P-1;j++) {
    indicesw[j-1] = left + j;
    double inter1 = mat_u - indicesw[j-1];
    if(antialiasing && scale<1)
      inter1 = inter1*scale;

    if(inter1 >= -1 && inter1<0) {
      weightw[j-1] = 1+ inter1;
    }
    else if(inter1>=0 && inter1<=1) {
      weightw[j-1] = 1-inter1;
    }
    else {
      weightw[j-1] = 0;
    }
    if(antialiasing && scale<1)
      weightw[j-1] *= scale;

    sum += weightw[j-1];
    indicesw[j-1] = min(width, max(1, indicesw[j-1]));
  }
  for(int j=1;j<P-1;j++) 
    weightw[j-1]/=sum;

  //Calculate mtxinter1. Each thread calculate one element in width
  //Each block is one element in new_w
  for(int j=tid;j<new_h;j+=blockDim.x) {
    out[j*new_w+ i_index] = 0.0;
    for(int k=1;k<P-1;k++)
      out[j*new_w+ i_index] += mtxinter1[indicesw[k-1] - 1 + width * j] * weightw[k-1];
  }
}
