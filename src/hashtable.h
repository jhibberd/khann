#define TABLE_SIZE 256
#define index(k) hash((unsigned char *) k) % TABLE_SIZE

void *hashtable_get(const char *k);
void hashtable_set(const char *k, void *obj);
void hashtable_destroy(void);

