The problem statement is a custom reader-writer lock with constraints:
1. At most 2 readers simultaneously
2. Exactly one writer at a time
3. No reader during writing
4. Readers are at a higher priority than the writers.

So, to approach the problem we firstly define our lock structure "pthread_lab1_lock_t" with fields:
1. active_readers → enforce max 2 readers
2. active_writer → ensure single writer
3. waiting_readers → needed for reader priority
4. waiting_writers → block writers when readers are waiting
5. mutex → protect shared lock state
6. condition variables (readers_cv and writers_cv) → block/wakeup threads

Shared variables (resources) among all functions:
int counter=0;
pthread_lab1_lock_t lock;

There are 50 threads(Thread ID: 0 to 49) in the problem. The even number threads acts as readers and the odd number threads acts as writers.
We needed to implement read_(lock/unlock), write_(lock/unlock), a thread_function that makes the even ID threads to read the counter and odd id threads to increment the counter. And the finally there is a main function implementing the complete workflow.
Firstly we did initialization with _init functions. 
Function Declaration:
1. void pthread_lab1_lock_init(pthread_lab1_lock_t *l);
2. void pthread_lab1_read_lock(pthread_lab1_lock_t *l);
3. void pthread_lab1_read_unlock(pthread_lab1_lock_t *l);
4. void pthread_lab1_write_lock(pthread_lab1_lock_t *l);
5. void pthread_lab1_write_unlock(pthread_lab1_lock_t *l);
6. void* thread_function(void* arg);

Logic outline for each function:
1. void pthread_lab1_lock_init(pthread_lab1_lock_t *l);
    Initializes all the memebrs of the lock struct.

2. void pthread_lab1_read_lock(pthread_lab1_lock_t *l);
    This function is called by reader thread before entering the critical section.
    Firstly the reader thread says "I want to read", so it locks the mutex and increment the number of waiting readers.
    Then it checks for its constraints to read (while loop):
    Reader cannot enter if:
    (i) a writer is active
    (ii) already 2 readers are there in the critical section
    The reader waits for time being
    Whenever the reading conditions meet, the reader thread decrements the number of waiting readers and increment active readers.
    Finally it unlocks mutex.

3. void pthread_lab1_read_unlock(pthread_lab1_lock_t *l);
    This function is called when the reader exits.
    It firstly locks the mutex, i.e. protect shared state. decrement the number of active readers as one reader had left.
    Then, it checks if there are no active readers (no need for checking the waiting readers, if there had been any they would have already entered the critical section and became active as there was space) and there are waiting writers, wakeup one writer (signal).
    else wake up all readers (up to 2 can enter the critical section at once).
    Finally unlock the mutex.

4. void pthread_lab1_write_lock(pthread_lab1_lock_t *l);
    This function is called by a writer thread before writing.
    The writer thread locks the mutex and increments the number of waiting_writers, waiting to write.
    The thread must waits if:
    (i) another writer is active
    (ii) readers are inside the critical section
    (iii) readers are in waiting state (reader priority)
    The writer sleeps for time being
    Whenever the writing conditions are met, the writer thread decrements the number of waitin_writers and increment the active_writer
    The writer enters the critical space.
    Finally unlock the mutex.

5. void pthread_lab1_write_unlock(pthread_lab1_lock_t *l);
    The function is called when writer exits. 
    It firstly locks the mutex and sets the active_writer as 0.
    Then it checks whether there are any waiting_readers, if so wake then up.
    Else the writer can go and write in the critical section.
    At last unlock the mutex.

6. void* thread_function(void* arg);
    It is the thread entry point. The function first checks if the thread has an even or odd id. 
    If the id is even it sends the thread to read the counter, i.e., first pthread_lab1_read_lock then critical section and finally pthread_lab1_read_unlock
    else it sends the thread to write the counter, i.e., pthread_lab1_write_lock, then critical section and finally pthread_lab1_write_unlock.

7. int main()
    The main functions firstly creates one array of 50 threads and another array for storing their thread IDs
    It initializes the lock by calling pthread_lab1_lock_init(&lock);
    Then it creates 50 threads using pthread_create(&thread[i], NULL, thread_function, &thread_id[i]); in a for loop
    Once the creation is completed the main thread calls pthread_join for all threads, thereby waiting for the completion for all threads.
    Finally, the last written value of counter is printed.

This complete workflow ensures that all the constraints of the problem are satisfied.


 