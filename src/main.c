/* Command line utility for performing neural network opertions.
 *
 * Usage: ./network -(v|t)
 */

main(int argx, char **argv) 
{
    switch (argv[1][1]) {
        case 't':
            train_network();
            break;
        case 'v':
            validate_network();
            break;
    }
}
