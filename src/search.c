#include "nsearch.h"

#include <string.h>

static void heap_push(HeapVec *h, HeapEnt e) {
    VEC_PUSH(*h, e);
    size_t i = h->n - 1;
    while (i) {
        size_t parent = (i - 1) / 2;
        if (h->p[parent].lbl <= e.lbl) break;
        h->p[i] = h->p[parent];
        i = parent;
    }
    h->p[i] = e;
}

static int heap_pop(HeapVec *h, HeapEnt *out) {
    if (!h->n) return 0;
    *out = h->p[0];
    HeapEnt last = h->p[--h->n];
    size_t i = 0;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= h->n) break;
        if (c + 1 < h->n && h->p[c + 1].lbl < h->p[c].lbl) c++;
        if (h->p[c].lbl >= last.lbl) break;
        h->p[i] = h->p[c];
        i = c;
    }
    if (h->n) h->p[i] = last;
    return 1;
}

void search_init(Search *S, const Graph *g, Alg alg, int p, int me) {
    memset(S, 0, sizeof *S);
    S->g = g;
    S->alg = alg;
    if (p > 1) {
        part_make(&S->part, g, p, me);
    } else {
        S->part = (Part){.p = 1, .me = 0, .bound = {0, g->n}};
    }
    S->key = malloc((size_t)g->n * sizeof(lbl_t));
    MPI_Type_contiguous((int)sizeof(Msg), MPI_BYTE, &S->msg_t);
    MPI_Type_commit(&S->msg_t);
}

/* Lower the label of v. Owned nodes are queued locally; foreign nodes are
 * buffered for their owner. key[] of a foreign node records the best label sent. */
static void offer(Search *S, uint32_t v, lbl_t c) {
    lbl_t old = S->key[v];
    if (c >= old) return;
    S->key[v] = c;
    int o = part_owner(&S->part, v);
    if (o != S->part.me) {
        VEC_PUSH(S->out[o], ((Msg){v, c}));
    } else if (S->alg == ALG_DIJKSTRA) {
        heap_push(&S->heap, (HeapEnt){c, v});
    } else if (old == LBL_INF) {
        VEC_PUSH(S->next, v);
    }
}

static void relax_node(Search *S, uint32_t u, lbl_t base) {
    const Graph *g = S->g;
    for (uint32_t i = g->xadj[u]; i < g->xadj[u + 1]; i++)
        offer(S, g->adj[i], base + LBL_STEP(S->alg == ALG_BFS ? 1 : g->w[i]));
}

static void dijkstra_drain(Search *S) {
    HeapEnt e;
    while (heap_pop(&S->heap, &e))
        if (e.lbl == S->key[e.node]) relax_node(S, e.node, e.lbl);
}

static void bfs_expand(Search *S) {
    for (size_t i = 0; i < S->cur.n; i++) relax_node(S, S->cur.p[i], S->key[S->cur.p[i]]);
}

static void swap_frontiers(Search *S) {
    U32Vec t = S->cur;
    S->cur = S->next;
    S->next = t;
    S->next.n = 0;
}

static void msgvec_reserve(MsgVec *v, size_t n) {
    if (v->cap < n) {
        v->cap = n;
        v->p = realloc(v->p, n * sizeof(Msg));
    }
}

/* One bulk-synchronous exchange. Returns 0 when no rank has anything left. */
static int exchange(Search *S) {
    int p = S->part.p, sc[MAXP], rc[MAXP], sd[MAXP], rd[MAXP];
    uint64_t local[2] = {0, S->next.n}, global[2];
    for (int r = 0; r < p; r++) {
        sc[r] = (int)S->out[r].n;
        local[0] += (uint64_t)sc[r];
    }
    double a = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double b = MPI_Wtime();
    MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(local, global, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    size_t stot = 0, rtot = 0;
    for (int r = 0; r < p; r++) {
        sd[r] = (int)stot; stot += (size_t)sc[r];
        rd[r] = (int)rtot; rtot += (size_t)rc[r];
    }
    if (global[0]) {
        msgvec_reserve(&S->sbuf, stot);
        msgvec_reserve(&S->rbuf, rtot);
        for (int r = 0; r < p; r++)
            if (sc[r]) memcpy(S->sbuf.p + sd[r], S->out[r].p, (size_t)sc[r] * sizeof(Msg));
        MPI_Alltoallv(S->sbuf.p, sc, sd, S->msg_t, S->rbuf.p, rc, rd, S->msg_t, MPI_COMM_WORLD);
    }
    double c = MPI_Wtime();
    S->st.wait += b - a;
    S->st.xfer += c - b;
    S->st.msgs += stot;
    for (int r = 0; r < p; r++) S->out[r].n = 0;
    for (size_t i = 0; i < rtot; i++) offer(S, S->rbuf.p[i].node, S->rbuf.p[i].lbl);
    return (global[0] + global[1]) > 0;
}

static void run_dijkstra(Search *S) {
    for (;;) {
        dijkstra_drain(S);
        S->st.steps++;
        if (S->part.p == 1 || !exchange(S)) return;
    }
}

static void run_bfs(Search *S) {
    swap_frontiers(S);
    for (;;) {
        bfs_expand(S);
        S->st.steps++;
        int more = S->part.p == 1 ? S->next.n > 0 : exchange(S);
        if (!more) return;
        swap_frontiers(S);
    }
}

void search_run(Search *S, const Workload *W, const uint32_t *ids, uint32_t cnt) {
    for (uint32_t v = 0; v < S->g->n; v++) S->key[v] = LBL_INF;
    S->heap.n = S->cur.n = S->next.n = 0;
    for (int r = 0; r < MAXP; r++) S->out[r].n = 0;
    for (uint32_t j = 0; j < cnt; j++) {
        uint32_t node = W->pool_node[ids[j]];
        if (part_owner(&S->part, node) == S->part.me) offer(S, node, LBL_MAKE(0, ids[j]));
    }
    if (S->alg == ALG_DIJKSTRA) run_dijkstra(S);
    else run_bfs(S);
}
