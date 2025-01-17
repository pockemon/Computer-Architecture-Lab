
#include "bits/stdc++.h"
#include "omp.h"
using namespace std;

void printHello(int threadId)
{
	printf("Hello from thread : %d\n",threadId);
}

int main()
{
	// create parallel region
	#pragma omp parallel
	{
		// get thread ID
		int thread_number = omp_get_thread_num();
		printHello(thread_number);
	}
}