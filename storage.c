#include <stdio.h>
#include "ann.h"

void load_weights_from_file(void);
void save_weights_to_file(void);

extern int topology[];
extern int num_layers;
extern float wgt[LAYERS][MAX_LAYER_SIZE][MAX_LAYER_SIZE];

void save_weights_to_file(void) 
{
    printf("Saving weights...\n");
    fflush(stdout);

    int l, j, k;
    float x;
    FILE *fp;

    fp = fopen("ann.data", "w");

    for (l = 1; l < num_layers; ++l)
        for (j = 0; j < topology[l]; ++j)
            for (k = 0; k < topology[l-1]; ++k) {
                x = wgt[l][j][k];
                fprintf(fp, "%f,", x);
            }

    fclose(fp);
}

void load_weights_from_file(void) 
{
    int l, j, k;
    float x;
    FILE *fp;

    fp = fopen("ann.data", "r");

    for (l = 1; l < num_layers; ++l)
        for (j = 0; j < topology[l]; ++j)
            for (k = 0; k < topology[l-1]; ++k) {
                fscanf(fp, "%f,", &x);
                wgt[l][j][k] = x;
            }

    fclose(fp);
}

