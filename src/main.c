/* Command line utility for performing neural network opertions.
 *
 * Usage: ./network -(v|t)
 */

#include "khann.h"

int main(int argx, char **argv) 
{
    char *nid;

    switch (argv[1][1]) {
        case 't':
            nid = argv[2];
            train_network(nid);
            break;
        case 'v':
            nid = argv[2];
            validate_network(nid);
            break;
    }
}
