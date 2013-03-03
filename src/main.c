/* Command line utility for performing neural network opertions.
 *
 * Usage: ./network -(v|t)
 */

#include "ann.h"

int main(int argx, char **argv) 
{
    switch (argv[1][1]) {
        case 't':
            train_network();
            break;
        case 'v':
            validate_network();
            break;
        /*case 's':
            time_network();
            break;*/
    }
}
