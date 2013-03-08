#include <stdlib.h>
#include "hashtable.h"

static void *table[TABLE_SIZE];
static unsigned long hash(unsigned char *str);

void *hashtable_get(const char *k)
{
    return table[index(k)];
}

void hashtable_set(const char *k, void *obj)
{
    table[index(k)] = obj;
}

void hashtable_destroy(void)
{
    int i;

    for (i = 0; i < TABLE_SIZE; ++i)
        if (table[i] != NULL)
            free(table[i]);
}

static unsigned long hash(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;

    return hash;
}

