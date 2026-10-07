#include "nsearch.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint32_t rng_below(uint64_t *s, uint32_t n) { return (uint32_t)(rng_next(s) % n); }

static void die(const char *msg, const char *path) {
    fprintf(stderr, "error: %s: %s\n", msg, path);
    exit(1);
}

static void must_read(void *dst, size_t size, size_t count, FILE *f, const char *path) {
    if (fread(dst, size, count, f) != count) die("short read", path);
}

Graph graph_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open graph", path);
    char magic[4];
    Graph g;
    must_read(magic, 1, 4, f, path);
    if (memcmp(magic, "NSG1", 4) != 0) die("bad graph magic", path);
    must_read(&g.n, 4, 1, f, path);
    must_read(&g.m, 4, 1, f, path);
    g.xadj = malloc((size_t)(g.n + 1) * 4);
    g.adj = malloc((size_t)g.m * 4);
    g.w = malloc((size_t)g.m * 4);
    must_read(g.xadj, 4, g.n + 1, f, path);
    must_read(g.adj, 4, g.m, f, path);
    must_read(g.w, 4, g.m, f, path);
    fclose(f);
    return g;
}

uint32_t *u32_file_load(const char *path, uint32_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open file", path);
    must_read(count, 4, 1, f, path);
    uint32_t *a = malloc((size_t)*count * 4);
    must_read(a, 4, *count, f, path);
    fclose(f);
    return a;
}

typedef struct { uint32_t a, b, w; } Edge;

static uint32_t uf_find(uint32_t *par, uint32_t x) {
    while (par[x] != x) x = par[x] = par[par[x]];
    return x;
}

/* Connected, planar, road-like grid: random spanning tree plus extra edges. */
Graph graph_synth(uint32_t target, uint64_t seed) {
    uint32_t W = (uint32_t)ceil(sqrt((double)target)), H = (target + W - 1) / W, n = W * H;
    size_t ncand = 0;
    Edge *cand = malloc(2 * (size_t)n * sizeof(Edge));
    uint64_t rs = seed;
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint32_t v = y * W + x;
            if (x + 1 < W) cand[ncand++] = (Edge){v, v + 1, 30 + rng_below(&rs, 271)};
            if (y + 1 < H) cand[ncand++] = (Edge){v, v + W, 30 + rng_below(&rs, 271)};
        }
    for (size_t i = ncand; i > 1; i--) {
        size_t j = rng_below(&rs, (uint32_t)i);
        Edge t = cand[i - 1]; cand[i - 1] = cand[j]; cand[j] = t;
    }
    uint32_t *par = malloc((size_t)n * 4);
    for (uint32_t i = 0; i < n; i++) par[i] = i;
    size_t nkeep = 0;
    for (size_t i = 0; i < ncand; i++) {
        uint32_t ra = uf_find(par, cand[i].a), rb = uf_find(par, cand[i].b);
        int keep = ra != rb || rng_below(&rs, 100) < 40;
        if (ra != rb) par[ra] = rb;
        if (keep) cand[nkeep++] = cand[i];
    }
    Graph g = {.n = n, .m = (uint32_t)(2 * nkeep)};
    g.xadj = calloc((size_t)n + 1, 4);
    g.adj = malloc((size_t)g.m * 4);
    g.w = malloc((size_t)g.m * 4);
    for (size_t i = 0; i < nkeep; i++) { g.xadj[cand[i].a + 1]++; g.xadj[cand[i].b + 1]++; }
    for (uint32_t v = 0; v < n; v++) g.xadj[v + 1] += g.xadj[v];
    uint32_t *fill = malloc((size_t)n * 4);
    memcpy(fill, g.xadj, (size_t)n * 4);
    for (size_t i = 0; i < nkeep; i++) {
        uint32_t a = cand[i].a, b = cand[i].b;
        g.adj[fill[a]] = b; g.w[fill[a]++] = cand[i].w;
        g.adj[fill[b]] = a; g.w[fill[b]++] = cand[i].w;
    }
    free(cand); free(par); free(fill);
    return g;
}

/* Contiguous node ranges holding roughly equal numbers of edges. */
void part_make(Part *P, const Graph *g, int p, int me) {
    P->p = p;
    P->me = me;
    P->bound[0] = 0;
    uint32_t v = 0;
    for (int r = 1; r < p; r++) {
        uint64_t target = (uint64_t)g->m * r / p;
        while (v < g->n && g->xadj[v] < target) v++;
        P->bound[r] = v;
    }
    P->bound[p] = g->n;
}

int part_owner(const Part *P, uint32_t v) {
    int r = 0;
    while (v >= P->bound[r + 1]) r++;
    return r;
}

double part_edge_cut(const Part *P, const Graph *g) {
    uint64_t cut = 0;
    for (uint32_t u = 0; u < g->n; u++) {
        int ou = part_owner(P, u);
        for (uint32_t i = g->xadj[u]; i < g->xadj[u + 1]; i++)
            cut += part_owner(P, g->adj[i]) != ou;
    }
    return g->m ? (double)cut / g->m : 0.0;
}
