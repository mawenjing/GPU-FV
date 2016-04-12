#include <iostream>
#include <omp.h>
using namespace std;
int man(int argc, char* argv[])
{
	omp_set_num_threads(16);
	int i;
	 #pragma omp parallel for private(i)
	for(  i=0;i<32;i++)
	  std::cout<<"I "<<omp_get_num_threads()<<endl;
return 0;
}
	
