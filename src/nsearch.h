#ifndef NSEARCH_H
#define NSEARCH_H

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* A label packs (distance << 32) | school pool index. Comparing labels orders
 * by distance, then by lowest school index, so ties break deterministically. */
typedef uint64_t lbl_t;
#define LBL_INF UINT64_MAX
#define LBL_MAKE(d, id) (((lbl_t)(d) << 32) | (uint32_t)(id))
#define LBL_DIST(l) ((uint32_t)((l) >> 32))
#define LBL_ID(l) ((uint32_t)(l))
#define LBL_STEP(w) ((lbl_t)(w) << 32)

#define MAXP 64

typedef enum { ALG_DIJKSTRA, ALG_BFS } Alg;
typedef enum { STRAT_SEQ, STRAT_SOURCE, STRAT_GRAPH, STRAT_SCENARIO } Strat;

typedef struct {
    uint32_t n, m;
    uint32_t *xadj, *adj, *w;
} Graph;

typedef struct {
    int p, me;
    uint32_t bound[MAXP + 1];
} Part;

typedef struct { uint32_t node; lbl_t lbl; } Msg;
typedef struct { lbl_t lbl; uint32_t node; } HeapEnt;

#define DEFVEC(name, T) typedef struct { T *p; size_t n, cap; } name
DEFVEC(U32Vec, uint32_t);
DEFVEC(MsgVec, Msg);
DEFVEC(HeapVec, HeapEnt);

#define VEC_PUSH(v, x)                                                        \
    do {                                                                      \
        if ((v).n == (v).cap) {                                               \
            (v).cap = (v).cap ? 2 * (v).cap : 1024;                           \
            (v).p = realloc((v).p, (v).cap * sizeof(*(v).p));                 \
        }                                                                     \
        (v).p[(v).n++] = (x);                                                 \
    } while (0)

typedef struct {
    double wait, xfer;
    uint64_t msgs;
    uint32_t steps;
} Stats;

typedef struct {
    uint32_t pool_n, k, q, B;
    uint32_t *pool_node, *queries;
    uint64_t seed;
} Workload;

typedef struct {
    const Graph *g;
    Alg alg;
    Part part;
    HeapVec heap;
    U32Vec cur, next;
    MsgVec out[MAXP], sbuf, rbuf;
    lbl_t *key;
    Stats st;
    MPI_Datatype msg_t;
} Search;

/* graph.c */
uint64_t rng_next(uint64_t *state);
uint32_t rng_below(uint64_t *state, uint32_t n);
Graph graph_load(const char *path);
Graph graph_synth(uint32_t target_nodes, uint64_t seed);
uint32_t *u32_file_load(const char *path, uint32_t *count);
void part_make(Part *P, const Graph *g, int p, int me);
int part_owner(const Part *P, uint32_t v);
double part_edge_cut(const Part *P, const Graph *g);

/* workload.c */
void workload_init(Workload *W, const Graph *g, uint32_t pool_n, uint32_t *pool_node,
                   uint32_t k, uint32_t q, uint32_t B, uint64_t seed);
void scenario_ids(const Workload *W, uint32_t s, uint32_t *ids);

/* search.c */
void search_init(Search *S, const Graph *g, Alg alg, int p, int me);
void search_run(Search *S, const Workload *W, const uint32_t *ids, uint32_t cnt);

/* strategies.c */
void strat_seq(Search *local, const Workload *W, lbl_t *res);
void strat_source(Search *local, const Workload *W, int rank, int p, lbl_t *res);
void strat_scenario(Search *local, const Workload *W, int rank, int p, lbl_t *res,
                    uint64_t *perr);
void strat_graph(Search *dist, const Workload *W, lbl_t *res, uint64_t *perr);

#endif
