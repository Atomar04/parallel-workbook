#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "cJSON.h"
#include "cJSON.c"

#define FILE_NAME "Movies_and_TV_5.json"
#define TABLE_SIZE 4000007 // a large prime number

// creating a hash table structure
typedef struct {
    char reviewerID[32];
    int count;
    long total_words;
    int used;
} ReviewerEntry;

ReviewerEntry table[TABLE_SIZE];

// using a classic djb2 hash function
unsigned long hash_func(const char *s) {
    unsigned long hash = 5381;
    int c;
    while ((c = *s++)) {
        hash = ((hash << 5) + hash) + c;
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
    strncpy(table[idx].reviewerID, id, sizeof(table[idx].reviewerID) - 1);
    table[idx].reviewerID[sizeof(table[idx].reviewerID) - 1] = '\0';
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
    // open the json file
    FILE *fp = fopen(FILE_NAME, "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    long total_reviews = 0;
    long elaborate_reviews = 0;
    long parse_errors = 0;

    struct timespec start, end;
    //start timer before starting reading
    clock_gettime(CLOCK_MONOTONIC, &start);
    // read the input file one JSON object (one review) per line
    while ((read = getline(&line, &len, fp)) != -1) {
        total_reviews++;
        // parse the current JSON line into a cJSON object tree
        cJSON *root = cJSON_Parse(line);
        if (!root) {
            parse_errors++;
            continue;
        }
        // extract the "reviewerID" and "reviewText" fields from the JSON object
        cJSON *rid = cJSON_GetObjectItemCaseSensitive(root, "reviewerID");
        cJSON *txt = cJSON_GetObjectItemCaseSensitive(root, "reviewText");

        // process only if both fields exist and are valid JSON strings
        if (cJSON_IsString(rid) && rid->valuestring &&
            cJSON_IsString(txt) && txt->valuestring) {
            // count the number of words in the reviewTest
            int words = count_words(txt->valuestring);
            // a review if considered elaborate if words in reviewText>=50
            // cheking if words >=50
            if (words >= 50) {
                elaborate_reviews++;
                // update the reviewer entry in the hash table
                // that is increment the elaborate review count and the number of words
                update_reviewer(rid->valuestring, words);
            }
        }
        // free memory allocated by cJSON for this parsed review object
        cJSON_Delete(root);
    }
    // free up the file pointer and line pointer
    fclose(fp);
    free(line);

    clock_gettime(CLOCK_MONOTONIC, &end);
    // output the reviewerID, elaborate review count and average review length
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

    return 0;
}
