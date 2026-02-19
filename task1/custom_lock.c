#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>

#define NTHREAD 50

typedef struct{
    int active_readers;
    int active_writer;
    int waiting_readers;
    int waiting_writers;

    pthread_mutex_t mutex;
    pthread_cond_t readers_cv;
    pthread_cond_t writers_cv;
}pthread_lab1_lock_t;

pthread_lab1_lock_t lock;
int counter=0;

void pthread_lab1_read_lock(pthread_lab1_lock_t *l);
void pthread_lab1_read_unlock(pthread_lab1_lock_t *l);
void pthread_lab1_write_lock(pthread_lab1_lock_t *l);
void pthread_lab1_write_unlock(pthread_lab1_lock_t *l);

void pthread_lab1_lock_init(pthread_lab1_lock_t *l){
    l->active_readers=l->active_writer=0;
    l->waiting_readers=l->waiting_writers=0;

    pthread_mutex_init(&(l->mutex), NULL);
    pthread_cond_init(&(l->readers_cv), NULL);
    pthread_cond_init(&(l->writers_cv), NULL);
}

//Read lock
void pthread_lab1_read_lock(pthread_lab1_lock_t *l){
    pthread_mutex_lock(&l->mutex);
    l->waiting_readers++;

    while(l->active_writer==1 || l->active_readers==2){
        pthread_cond_wait(&(l->readers_cv), &(l->mutex));
    }
    l->waiting_readers--;
    l->active_readers++;
    pthread_mutex_unlock(&(l->mutex));

}

//Read unlock
void pthread_lab1_read_unlock(pthread_lab1_lock_t *l){
    pthread_mutex_lock(&l->mutex);
    l->active_readers--;
    if(l->active_readers==0 && l->waiting_writers>0){
        pthread_cond_signal(&l->writers_cv);
    }
    else{
        pthread_cond_broadcast(&l->readers_cv);
    }
    pthread_mutex_unlock(&l->mutex);
}

//Write lock
void pthread_lab1_write_lock(pthread_lab1_lock_t *l){
    pthread_mutex_lock(&l->mutex);
    l->waiting_writers++;
    while(l->active_writer==1|| l->active_readers>0 ||l->waiting_readers>0){
        pthread_cond_wait(&l->writers_cv, &l->mutex);
    }
    l->waiting_writers--;
    l->active_writer=1;
    pthread_mutex_unlock(&l->mutex);
}

//write unlock
void pthread_lab1_write_unlock(pthread_lab1_lock_t *l){
    pthread_mutex_lock(&l->mutex);
    l->active_writer=0;
    if(l->waiting_readers>0){
        pthread_cond_broadcast(&l->readers_cv);
    }
    else{
        pthread_cond_signal(&l->writers_cv);
    }
    pthread_mutex_unlock(&l->mutex);
}


void* thread_function(void* arg){
    int thread_id= *(int*)arg;
    if((thread_id % 2)==0){
        //Reader thread
        pthread_lab1_read_lock(&lock);
        printf("Reader thread with ID %d reads counter: %d\n", thread_id, counter);
        pthread_lab1_read_unlock(&lock);
    }
    else{
        //Writer thread
        pthread_lab1_write_lock(&lock);
        counter++;
        printf("Writer thread with ID %d writes counter: %d\n", thread_id, counter);
        pthread_lab1_write_unlock(&lock);
    }
    return NULL;
}

int main(){
    pthread_t thread[NTHREAD]; 
    int thread_id[NTHREAD];
    pthread_lab1_lock_init(&lock);
    for(int i=0; i<NTHREAD; i++){
        thread_id[i]=i; // 0 to 49
        printf("Creating thread %d\n", i);
        pthread_create(&thread[i], NULL, thread_function, &thread_id[i]);
    }
    printf("Thread creation completed.\n");
    for(int i=0; i<NTHREAD; i++){
        pthread_join(thread[i], NULL);
    }
    printf("Final value of counter: %d\n", counter); 
    return 0;
}


