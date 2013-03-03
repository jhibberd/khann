#include <stdio.h>
#include "mongo.h"

/* gcc --std=c99 -I/usr/local/include -L/usr/local/lib main.c -lmongoc */
/* TODO(jhibberd) Remove training files from data? Also delete old gen gen
 * files */
/* TODO(jhibberd) Modify C neural network to read training set from mongoDB,
 * rather than flat file */
/* Rename "data" dir to "weights" and remove ".weights" from filename */

#define IV_SIZE 3
#define OV_SIZE 2

int main() {

    /* Establish connection */
    mongo conn[1];
    int status = mongo_client( conn, "127.0.0.1", 27017 );
    if (status != MONGO_OK) {
        switch (conn->err) {
            case MONGO_CONN_NO_SOCKET:  
                printf("no socket\n"); 
                return 1;
            case MONGO_CONN_FAIL:       
                printf("connection failed\n"); 
                return 1;
            case MONGO_CONN_NOT_MASTER: 
                printf( "not master\n" ); 
                return 1;
        }
    }

    bson b[1];
    
    bson_init(b);

    bson_append_string(b, "_id", "testy");
    bson_append_start_array(b, "data");
    bson_append_double(b, "0", 1.3);
    bson_append_double(b, "1", 2.3);
    bson_append_finish_array(b);

    bson_finish(b);
    mongo_insert(conn, "khann__system.weights", b, NULL);

    bson_destroy(b);
    mongo_destroy(conn);
    exit(EXIT_SUCCESS);


    float *iv, *ov;

    iv = (float *)malloc(IV_SIZE * sizeof(float));
    ov = (float *)malloc(OV_SIZE * sizeof(float));


    double n = mongo_count(conn, "khann_alphanum", "training", NULL);
    printf("==%f==", n);

    /* Iterate over docs */
    mongo_cursor cursor[1];
    mongo_cursor_init(cursor, conn, "khann_alphanum.training");

    while (mongo_cursor_next(cursor) == MONGO_OK) {
        float *p;

        bson_iterator i[1], sub[1];
        bson *b = &cursor->current;
        bson_iterator_init(i, b);

        bson_find(i, b, "iv");
        bson_iterator_subiterator(i, sub);
        p = iv;
        while (bson_iterator_next(sub) != BSON_EOO)
            *p++ = (float)bson_iterator_double(sub);

        bson_find(i, b, "ov");
        bson_iterator_subiterator(i, sub);
        p = ov;
        while (bson_iterator_next(sub) != BSON_EOO)
            *p++ = (float)bson_iterator_double(sub);
    }

    int i;
    printf("IV\n");
    for (i = 0; i < IV_SIZE; ++i)
        printf("%f,", iv[i]);
    printf("\nOV\n");
    for (i = 0; i < OV_SIZE; ++i)
        printf("%f,", ov[i]);

    mongo_cursor_destroy( cursor );
    mongo_destroy( conn );
    return 0;
}
