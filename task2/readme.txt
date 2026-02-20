The problem is a 2D heat equation on a square metal plate.
Each row can be computed independently within the same timestep.

1. Firstly, modifying file (1) TempGrid_HW3.c -> openmp_sol1_2022B4A70596P.c
    Logic:
        Compute new through parallel for
        copy new -> old through parallel for
        single thread prints
        next time step 

    All the variables and initialization is same as the original code. 
    The difference is in the way parallelization is done. 
    OpenMP uses:
        #pragma omp parallel num_threads(NumThreads)
        {
            ....
        }
    this is equivalent of pthread_create, thread function, pthread_join
    Unlike original file, we don't need to divide blocks for the threads. The openMP directive can take care of this automatically.
    Using: #pragma omp for
    Once the new temperature grid has been computed, we need to copy it to the old grid. All threads must finish their rows computation before continuing.
    We use:
        #pragma omp for 
    for copying, openMP for has an implicit barrier at its end, so once all threads are done only then next code outside for is executed 
    Now to copy wee again use omp for, it makes the loop faster 
    is used. We put a barrier after that, so that the copying is done before the threads proceed to next timesteps.
    Then a single thread prints the temperatures to avoid console output.
    Using: 
        #pragma omp single
        {
            ....
        }

2. Modifying file (2) HW3_1.c -> openmp_sol2_2022B4A70596P.c
    Now instead of copying the complete grid every time, we swap pointers to the grid. This reduces the huge copying overhead.
    Similar to HW3_1.c we define:
        float (*old)[Y_SIZE];
        float (*new)[Y_SIZE];
    The openMP directives are exactly same as in openmp_sol1_2022B4A70596P.c
    The only difference occurs at:
        //swapping pointers
        #pragma omp single
        {
            float (*temp)[Y_SIZE]=old;
            old=new;
            new=temp;
        }

3. Modifying file (3) HW3_2.c -> openmp_sol3_2022B4A70596P.c
    This file uses the swapping pointers approach along wth a computational efficiency.
    As the grid is symmetrical, temperature(j,k)==temperature(X_SIZE-1-j, k)
    So we can compute the left half and mirror it to right half.
    So instead of computing 1000x1000 values, we are essentially computing 500*1000 values
    This is done in the loop:
        for(int j=1; j<=halfX; j++){
            for(int k=1; k<=Y_SIZE-1; k++){
                if(j>0 && j<X_SIZE-1){
                    float val= old[j][k]+ Cx*(old[j+1][k]+old[j-1][k]-2*old[j][k])+ Cy*(old[j][k+1]+old[j][k-1]-2*old[j][k]);
                    new[j][k]=val;
                    new[X_SIZE-1-j][k]= val;
                }
            }
        }
    
    After this, pointer swapping is done in the exact same way ass the lasst file.
    Observation:
    The symmetry optimization changes the numerical solution because the explicit finite difference heat equation requires local neighbor evaluation. Mirroring the updated value enforces an artificial constraint that reduces effective diffusion, producing a physically incorrect solution.

Output:
{File Name: Static Schedule time,  Dynamic Schedule time, Guided Schedule time}
{openmp_sol1_2022B4A70596P.c: 2.269233019 sec, 4.204496124 sec, 2.278789018 sec}
{openmp_sol2_2022B4A70596P.c: 2.163639429 sec, 2.899663278 sec, 2.149923273 sec}
{openmp_sol3_2022B4A70596P.c: 1.184653899 sec, 1.630271171 sec, 1.201278312 sec}