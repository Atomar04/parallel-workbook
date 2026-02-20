//(3)
//modifying HW3_2.c

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define X_SIZE 1000
#define Y_SIZE 1000
#define Cx 0.125
#define Cy 0.11
#define CMin 200
#define CMax 800
#define TIMESTEPS 4000
#define PRINT_STEPS 200

float (*old)[Y_SIZE];
float (*new)[Y_SIZE];

int main(int argc, char** argv){

    if(argc < 2){
        fprintf(stderr,"Usage: %s <numthreads>\n",argv[0]);
        exit(1);
    }

    int NumThreads = atoi(argv[1]);

    old = malloc(sizeof(float[X_SIZE][Y_SIZE]));
    new = malloc(sizeof(float[X_SIZE][Y_SIZE]));
    // Initializing the grid
    for (int x = 0; x < X_SIZE; x++) {
        for (int y = 0; y < Y_SIZE; y++) {
            old[x][y] = (x >= CMin-1 && x <= CMax-1 && y >= CMin-1 && y <= CMax-1) ? 500.0 : 0.0;
        }
    }
    int halfX = X_SIZE/2;
    struct timespec StartTime, EndTime;
    clock_gettime(CLOCK_REALTIME, &StartTime);

#pragma omp parallel num_threads(NumThreads)
{
    for(int block=1; block<=TIMESTEPS; block++){
        // compute only half and mirror
        #pragma omp for schedule(runtime)
        for(int j=1; j<=halfX; j++){
            for(int k=1; k<=Y_SIZE-1; k++){
                if(j>0 && j<X_SIZE-1){
                    new[j][k]= old[j][k]+ Cx*(old[j+1][k]+old[j-1][k]-2*old[j][k])+ Cy*(old[j][k+1]+old[j][k-1]-2*old[j][k]);
                    new[X_SIZE-1-j][k]= new[j][k];
                }
            }
        }

        //swapping pointers
        #pragma omp single
        {
            float (*temp)[Y_SIZE]=old;
            old=new;
            new=temp;
        }

        #pragma omp single
        {
            if(block%PRINT_STEPS==0){
                printf("\nThe Temperature values at points:[1,1]=%f [150,150]=%f [400,400]=%f [500,500]=%f [750,750]=%f [900,900]=%f",old[1][1], old[150][150], old[400][400], old[500][500], old[750][750], old[900][900]);
            }
        }
    }
}
clock_gettime(CLOCK_REALTIME, &EndTime);
    unsigned long long int runtime = 1000000000 * (EndTime.tv_sec - StartTime.tv_sec) + EndTime.tv_nsec - StartTime.tv_nsec;
    printf("\nTime = %llu nanoseconds (%.9f sec)\n", runtime, runtime / 1000000000.0);
    free(old);
    free(new);
    return 0;
}