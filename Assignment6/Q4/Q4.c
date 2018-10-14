#include <stdio.h>
#include <mpi.h>
long long num_steps = 100000000;
int main(int argc, char const *argv[])
{
	MPI_Init(NULL,NULL);
	int rank,world_size;

	double pi,time,sum=0.0,x,t_sum=0.0;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&world_size);

	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime();

	MPI_Bcast(&num_steps,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

	for(int i=rank;i<num_steps;i+=world_size)
	{
		x = (i+0.5)/num_steps;
		t_sum = t_sum + 4.0/(1.0 + x*x);
	}
	MPI_Reduce(&t_sum,&sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime() - time;

	
	if(rank == 0)
	{
		pi = sum/num_steps;
		printf("Value of pi calculated is %lf in %lf ms\n", pi,time*100);
	}


	return 0;
}
