#include <stdio.h>
#include <mpi.h>
#define BufferSize 1000
char buffer[BufferSize];

void fill_buffer_variables(char *c,int iA[2],float fA[4],char key)
{
	*c = key;
	for(int i=0;i<4;++i)
	{
		if(i<2)
		{
			iA[i] = key + i;
		}
		fA[i] = key + i;
	}
}
int main(int argc, char const *argv[])
{
	int rank,world_size,namelen;
	char name[30];

	MPI_Init(NULL,NULL);

	MPI_Comm_size(MPI_COMM_WORLD,&world_size);

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	MPI_Get_processor_name(name,&namelen);


	char c;
	int iA[2];
	float fA[4];

	if(rank == 0)
	{
		fill_buffer_variables(&c,iA,fA,'A');

		printf("Packed in Rank : %d Values : %c - %d %d - %f %f %f %f\n",rank,c,iA[0],iA[1],fA[0],fA[1],fA[2],fA[3]);
		
		int position = 0;
		MPI_Pack(&c,1,MPI_CHAR,buffer,100,&position,MPI_COMM_WORLD);
		MPI_Pack(iA,2,MPI_INT,buffer,100,&position,MPI_COMM_WORLD);
		MPI_Pack(fA,4,MPI_FLOAT,buffer,100,&position,MPI_COMM_WORLD);

		MPI_Bcast(buffer, BufferSize, MPI_PACKED, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Bcast(buffer, BufferSize, MPI_PACKED, 0, MPI_COMM_WORLD);

		int position = 0;
		MPI_Unpack(buffer,BufferSize,&position,&c,1,MPI_CHAR,MPI_COMM_WORLD);
		MPI_Unpack(buffer,BufferSize,&position,iA,2,MPI_INT,MPI_COMM_WORLD);
		MPI_Unpack(buffer,BufferSize,&position,fA,4,MPI_FLOAT,MPI_COMM_WORLD);

		printf("Unpacked - Rank : %d Values : %c - %d %d - %f %f %f %f\n",rank,c,iA[0],iA[1],fA[0],fA[1],fA[2],fA[3]);
		
	}
	MPI_Finalize();

	return 0;
}
