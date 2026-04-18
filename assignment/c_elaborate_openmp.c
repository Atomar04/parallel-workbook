#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>
#include "cJSON.h"
#include "cJSON.c"

#define FILE_NAME "Movies_and_TV_5.json"
#define MAX_ID_LEN 64
#define TABLE_SIZE 4000007

// structure for one reviewer entry in final hash table
typedef struct {
    char reviewerID[MAX_ID_LEN];
    int count;
    long total_words;
    int used;
} ReviewerEntry;

// structure for intermediate parallel results
typedef struct {
    char reviewerID[MAX_ID_LEN];
    int words;
    int valid;
} PartialResult;

// global final hash table
static ReviewerEntry table[TABLE_SIZE];


// standard djb2 hash functions for reviewer IDs
unsigned long hash_func(const char *s) {
    unsigned long hash = 5381;
    int c;
    while ((c = *s++)) {
        hash = ((hash << 5) + hash) + c;   //hash = hash*33 + c 
    }
    return hash % TABLE_SIZE;
}

// function for incrementing review count of a reviewerID in hash table
void update_reviewer(const char *id, int words) {
    unsigned long idx = hash_func(id);
    while (table[idx].used) {
        if (strcmp(table[idx].reviewerID, id) == 0) {
            table[idx].count++;
            table[idx].total_words += words;
            return;
        }
        idx = (idx + 1) % TABLE_SIZE;
    }

    table[idx].used = 1;
    strncpy(table[idx].reviewerID, id, MAX_ID_LEN - 1);
    table[idx].reviewerID[MAX_ID_LEN - 1] = '\0';
    table[idx].count = 1;
    table[idx].total_words = words;
}

// function for counting the words in the reviewText
// needed for checking if len(text)>=50 and for average review length;
int count_words(const char *text) {
    int count = 0, in_word = 0;
    while (*text) {
        if (isspace((unsigned char)*text)) {
            in_word = 0;
        } else if (!in_word) {
            count++;
            in_word = 1;
        }
        text++;
    }
    return count;
}

int main() {
    FILE *fp = fopen(FILE_NAME, "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    char **lines = NULL;
    size_t n = 0, cap = 0;
    char *line = NULL;
    size_t len = 0;


    struct timespec start, end;
    // start timer: includes file reading+ processing+ merge
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Read all lines into memory first 
    while (getline(&line, &len, fp) != -1) {
        if (n == cap) {
            cap = (cap == 0) ? 1024 : cap * 2;
            char **tmp = realloc(lines, cap * sizeof(char *));
            if (!tmp) {
                perror("realloc failed");
                fclose(fp);
                free(line);
                return 1;
            }
            lines = tmp;
        }
        lines[n] = strdup(line);
        if (!lines[n]) {
            perror("strdup failed");
            fclose(fp);
            free(line);
            return 1;
        }
        n++;
    }
    fclose(fp);
    free(line);

    // allocate intermediate results array
    // results[i] stores the output of processing lines[i]
    PartialResult *results = calloc(n, sizeof(PartialResult));
    if (!results) {
        perror("calloc failed");
        return 1;
    }

    long total_reviews = (long)n;
    long elaborate_reviews = 0;
    long parse_errors = 0;

    
    // Parallel review scanning loop:
    // each iteration handles one review line, only elaborate reviews (>= 50 words) are recorded
    // results[i] is private to iteration i, so no race occurs there
    // scalar counters are combined safely using OpenMP reduction
     
    #pragma omp parallel for schedule(static) reduction(+:elaborate_reviews, parse_errors)
    for (long i = 0; i < (long)n; i++) {
        cJSON *root = cJSON_Parse(lines[i]);
        if (!root) {
            parse_errors++;
            continue;
        }

        cJSON *rid = cJSON_GetObjectItemCaseSensitive(root, "reviewerID");
        cJSON *txt = cJSON_GetObjectItemCaseSensitive(root, "reviewText");

        if (cJSON_IsString(rid) && rid->valuestring &&
            cJSON_IsString(txt) && txt->valuestring) {

            int words = count_words(txt->valuestring);

            if (words >= 50) {
                results[i].valid = 1;
                results[i].words = words;
                strncpy(results[i].reviewerID, rid->valuestring, MAX_ID_LEN - 1);
                results[i].reviewerID[MAX_ID_LEN - 1] = '\0';
                elaborate_reviews++;
            }
        }

        cJSON_Delete(root);
    }

    /* 
    Sequential merge into hash table
    This avoids synchronization overhead inside the parallel region.
    */
    for (size_t i = 0; i < n; i++) {
        if (results[i].valid) {
            update_reviewer(results[i].reviewerID, results[i].words);
        }
    }

    // stopping timer after aggregation
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Print reviewerID, with number of elaborate review and average review length
    printf("Elaborate Reviewers\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (table[i].used && table[i].count >= 5) {
            double avg = (double)table[i].total_words / table[i].count;
            printf("ReviewerID: %s, Count: %d, Average Review Length: %.2f words\n",
                   table[i].reviewerID, table[i].count, avg);
        }
    }

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nStatistics\n");
    printf("Total reviews processed: %ld\n", total_reviews);
    printf("Elaborate reviews: %ld\n", elaborate_reviews);
    printf("Parse errors: %ld\n", parse_errors);
    printf("Execution time: %.6f seconds\n", elapsed);

    for (size_t i = 0; i < n; i++) free(lines[i]);
    free(lines);
    free(results);

    return 0;
}
