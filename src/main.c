#include "nsearch.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    const char *graph, *schools, *label, *dump;
    uint32_t synth, pool, k, q, B, trials;
    uint64_t seed;
    Alg alg;
    Strat strat;
    int verify, header;
} Opts;

static const char *ALG_NAME[] = {"dijkstra", "bfs"};
static const char *STRAT_NAME[] = {"seq", "source", "graph", "scenario"};

static void usage(void) {
    fprintf(stderr,
            "usage: nsearch (--graph G.bin --schools S.bin | --synth N --pool P)\n"
            "  --alg dijkstra|bfs  --strategy seq|source|graph|scenario\n"
            "  [--k K] [--q Q] [--B B] [--trials T] [--seed S] [--label L]\n"
            "  [--verify] [--header] [--dump FILE]\n");
    exit(1);
}

static int pick(const char *s, const char **names, int n) {
    for (int i = 0; i < n; i++)
        if (!strcmp(s, names[i])) return i;
    usage();
    return -1;
}

static Opts parse(int argc, char **argv) {
    Opts o = {.label = "run", .k = 100, .q = 1000, .B = 1, .trials = 5, .seed = 42, .pool = 0};
    int have_alg = 0, have_strat = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "--verify")) o.verify = 1;
        else if (!strcmp(a, "--header")) o.header = 1;
        else if (!more) usage();
        else if (!strcmp(a, "--graph")) o.graph = argv[++i];
        else if (!strcmp(a, "--schools")) o.schools = argv[++i];
        else if (!strcmp(a, "--label")) o.label = argv[++i];
        else if (!strcmp(a, "--dump")) o.dump = argv[++i];
        else if (!strcmp(a, "--synth")) o.synth = (uint32_t)atol(argv[++i]);
        else if (!strcmp(a, "--pool")) o.pool = (uint32_t)atol(argv[++i]);
        else if (!strcmp(a, "--k")) o.k = (uint32_t)atol(argv[++i]);
        else if (!strcmp(a, "--q")) o.q = (uint32_t)atol(argv[++i]);
        else if (!strcmp(a, "--B")) o.B = (uint32_t)atol(argv[++i]);
        else if (!strcmp(a, "--trials")) o.trials = (uint32_t)atol(argv[++i]);
        else if (!strcmp(a, "--seed")) o.seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(a, "--alg")) { o.alg = (Alg)pick(argv[++i], ALG_NAME, 2); have_alg = 1; }
        else if (!strcmp(a, "--strategy")) { o.strat = (Strat)pick(argv[++i], STRAT_NAME, 4); have_strat = 1; }
        else usage();
    }
    if (!have_alg || !have_strat || (!o.synth && !(o.graph && o.schools)) || !o.B || !o.trials) usage();
    return o;
}

static uint32_t *synth_pool(const Graph *g, uint32_t pool, uint64_t seed) {
    uint32_t *perm = malloc((size_t)g->n * 4), *out = malloc((size_t)pool * 4);
    for (uint32_t i = 0; i < g->n; i++) perm[i] = i;
    uint64_t rs = seed ^ 0xABCDEFull;
    for (uint32_t i = 0; i < pool; i++) {
        uint32_t j = i + rng_below(&rs, g->n - i);
        uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        out[i] = perm[i];
    }
    free(perm);
    return out;
}

static void dump_origins(const char *path, const Workload *W, const lbl_t *res) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fprintf(f, "qidx,node,school_idx,dist\n");
    for (uint32_t j = 0; j < W->q; j++)
        fprintf(f, "%u,%u,%u,%u\n", j, W->queries[j], LBL_ID(res[j]), LBL_DIST(res[j]));
    fclose(f);
}

static uint64_t mismatches(const Graph *g, const Workload *W, Alg alg, const lbl_t *res) {
    Search ref;
    search_init(&ref, g, alg, 1, 0);
    size_t total = (size_t)W->B * W->q;
    lbl_t *want = malloc(total * sizeof(lbl_t));
    strat_seq(&ref, W, want);
    uint64_t bad = 0;
    for (size_t i = 0; i < total; i++) bad += want[i] != res[i];
    free(want);
    return bad;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    Opts o = parse(argc, argv);
    if (p > MAXP || (o.strat == STRAT_SEQ && p != 1)) {
        if (!rank) fprintf(stderr, "seq strategy requires exactly 1 process (max %d)\n", MAXP);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Graph g = o.synth ? graph_synth(o.synth, o.seed) : graph_load(o.graph);
    uint32_t pool_n, *pool_node;
    if (o.synth) {
        pool_n = o.pool ? o.pool : (g.n / 20 > 50 ? g.n / 20 : 50);
        pool_node = synth_pool(&g, pool_n, o.seed);
    } else {
        pool_node = u32_file_load(o.schools, &pool_n);
    }
    Workload W;
    workload_init(&W, &g, pool_n, pool_node, o.k, o.q, o.B, o.seed);

    Search S;
    int distributed = o.strat == STRAT_GRAPH;
    search_init(&S, &g, o.alg, distributed ? p : 1, distributed ? rank : 0);
    double cut = distributed ? part_edge_cut(&S.part, &g) : 0.0;

    size_t total = (size_t)W.B * W.q;
    lbl_t *res = rank == 0 ? malloc(total * sizeof(lbl_t)) : NULL;
    int failed = 0;
    if (o.header && rank == 0)
        printf("label,n,m,alg,strategy,p,k,q,B,pool,trial,wall,comp_min,comp_max,comp_mean,"
               "wait_mean,xfer_mean,msgs,steps,edge_cut,verified\n");

    for (uint32_t trial = 0; trial <= o.trials; trial++) {
        memset(&S.st, 0, sizeof S.st);
        uint64_t perr = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        switch (o.strat) {
        case STRAT_SEQ: strat_seq(&S, &W, res); break;
        case STRAT_SOURCE: strat_source(&S, &W, rank, p, res); break;
        case STRAT_GRAPH: strat_graph(&S, &W, res, &perr); break;
        case STRAT_SCENARIO: strat_scenario(&S, &W, rank, p, res, &perr); break;
        }
        double wall = MPI_Wtime() - t0;

        double mine[4] = {wall, wall - S.st.wait - S.st.xfer, S.st.wait, S.st.xfer};
        double mx[4], mn[4], sum[4];
        MPI_Reduce(mine, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(mine, mn, 4, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(mine, sum, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        uint64_t msgs = 0;
        MPI_Reduce(&S.st.msgs, &msgs, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            int verified = -1;
            if (o.verify && trial == 0 && o.strat != STRAT_SEQ)
                verified = (mismatches(&g, &W, o.alg, res) + perr) == 0;
            failed |= verified == 0;
            printf("%s,%u,%u,%s,%s,%d,%u,%u,%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%u,%.4f,%d\n",
                   o.label, g.n, g.m, ALG_NAME[o.alg], STRAT_NAME[o.strat], p, W.k, W.q, W.B, pool_n,
                   trial, mx[0], mn[1], mx[1], sum[1] / p, sum[2] / p, sum[3] / p,
                   (unsigned long long)msgs, S.st.steps, cut, verified);
            fflush(stdout);
        }
    }
    if (rank == 0 && o.dump && o.strat == STRAT_SEQ) dump_origins(o.dump, &W, res);
    MPI_Finalize();
    return failed ? 2 : 0;
}
