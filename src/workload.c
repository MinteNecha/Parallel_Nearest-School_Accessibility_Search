#include "nsearch.h"

#include <string.h>

/* Query origins are sampled once (seed fixed) and shared by every scenario. */
void workload_init(Workload *W, const Graph *g, uint32_t pool_n, uint32_t *pool_node,
                   uint32_t k, uint32_t q, uint32_t B, uint64_t seed) {
    W->pool_n = pool_n;
    W->pool_node = pool_node;
    W->k = k > pool_n ? pool_n : k;
    W->q = q > g->n ? g->n : q;
    W->B = B;
    W->seed = seed;
    uint32_t *perm = malloc((size_t)g->n * 4);
    for (uint32_t i = 0; i < g->n; i++) perm[i] = i;
    uint64_t rs = seed;
    W->queries = malloc((size_t)W->q * 4);
    for (uint32_t i = 0; i < W->q; i++) {
        uint32_t j = i + rng_below(&rs, g->n - i);
        uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        W->queries[i] = perm[i];
    }
    free(perm);
}

/* Scenario s is a random k-subset of the school pool, identical on every rank. */
void scenario_ids(const Workload *W, uint32_t s, uint32_t *ids) {
    uint32_t *perm = malloc((size_t)W->pool_n * 4);
    for (uint32_t i = 0; i < W->pool_n; i++) perm[i] = i;
    uint64_t rs = W->seed * 1000003ull + s + 1;
    for (uint32_t i = 0; i < W->k; i++) {
        uint32_t j = i + rng_below(&rs, W->pool_n - i);
        uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    memcpy(ids, perm, (size_t)W->k * 4);
    free(perm);
}
