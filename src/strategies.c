#include "nsearch.h"

#include <string.h>

/* Results are res[s * q + j]: packed label of query j in scenario s (root only). */

static void extract(const Search *S, const Workload *W, lbl_t *dst) {
    for (uint32_t j = 0; j < W->q; j++) dst[j] = S->key[W->queries[j]];
}

static uint32_t scenario_count(uint32_t B, int rank, int p) {
    return (uint32_t)rank < B ? (B - (uint32_t)rank + (uint32_t)p - 1) / (uint32_t)p : 0;
}

static void gather_displs(const int *counts, int *displs, int p) {
    int off = 0;
    for (int r = 0; r < p; r++) { displs[r] = off; off += counts[r]; }
}

static uint64_t count_unplaced(const uint8_t *placed, size_t n) {
    uint64_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += placed[i] != 1;
    return bad;
}

static void timed_xfer(Search *S, double t0) { S->st.xfer += MPI_Wtime() - t0; }

void strat_seq(Search *local, const Workload *W, lbl_t *res) {
    uint32_t *ids = malloc((size_t)W->k * 4);
    for (uint32_t s = 0; s < W->B; s++) {
        scenario_ids(W, s, ids);
        search_run(local, W, ids, W->k);
        extract(local, W, res + (size_t)s * W->q);
    }
    free(ids);
}

/* Schools of every scenario are dealt round-robin to ranks; label arrays are
 * merged with one MPI_MIN reduction over the query entries only. */
void strat_source(Search *local, const Workload *W, int rank, int p, lbl_t *res) {
    size_t total = (size_t)W->B * W->q;
    uint32_t *ids = malloc((size_t)W->k * 4), *sub = malloc((size_t)W->k * 4);
    lbl_t *mine = malloc(total * sizeof(lbl_t));
    for (uint32_t s = 0; s < W->B; s++) {
        scenario_ids(W, s, ids);
        uint32_t cnt = 0;
        for (uint32_t j = (uint32_t)rank; j < W->k; j += (uint32_t)p) sub[cnt++] = ids[j];
        search_run(local, W, sub, cnt);
        extract(local, W, mine + (size_t)s * W->q);
    }
    double t0 = MPI_Wtime();
    MPI_Reduce(mine, res, (int)total, MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
    timed_xfer(local, t0);
    local->st.msgs += total;
    free(ids); free(sub); free(mine);
}

/* Whole scenarios are dealt round-robin; ranks may own different numbers of
 * scenarios, so results are collected with MPI_Gatherv and placed by index. */
void strat_scenario(Search *local, const Workload *W, int rank, int p, lbl_t *res,
                    uint64_t *perr) {
    uint32_t mine_n = scenario_count(W->B, rank, p);
    uint32_t *ids = malloc((size_t)W->k * 4);
    lbl_t *mine = malloc((size_t)(mine_n ? mine_n : 1) * W->q * sizeof(lbl_t));
    for (uint32_t i = 0; i < mine_n; i++) {
        scenario_ids(W, (uint32_t)rank + i * (uint32_t)p, ids);
        search_run(local, W, ids, W->k);
        extract(local, W, mine + (size_t)i * W->q);
    }
    int counts[MAXP], displs[MAXP], my_count = (int)(mine_n * W->q);
    lbl_t *all = NULL;
    double t0 = MPI_Wtime();
    MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        gather_displs(counts, displs, p);
        all = malloc((size_t)(displs[p - 1] + counts[p - 1] + 1) * sizeof(lbl_t));
    }
    MPI_Gatherv(mine, my_count, MPI_UINT64_T, all, counts, displs, MPI_UINT64_T, 0,
                MPI_COMM_WORLD);
    timed_xfer(local, t0);
    local->st.msgs += (uint64_t)my_count;
    if (rank == 0) {
        size_t total = (size_t)W->B * W->q;
        uint8_t *placed = calloc(total, 1);
        for (int r = 0; r < p; r++)
            for (uint32_t i = 0; W->q && i < (uint32_t)counts[r] / W->q; i++) {
                size_t s = (size_t)r + (size_t)i * (size_t)p;
                if (s >= W->B) { (*perr)++; continue; }
                memcpy(res + s * W->q, all + displs[r] + (size_t)i * W->q, (size_t)W->q * sizeof(lbl_t));
                for (uint32_t j = 0; j < W->q; j++) placed[s * W->q + j]++;
            }
        *perr += count_unplaced(placed, total);
        free(placed); free(all);
    }
    free(ids); free(mine);
}

/* Every scenario is searched cooperatively on the partitioned graph. Each rank
 * answers the queries whose origin it owns; counts differ per rank, so answers
 * are collected with MPI_Gatherv together with their query indices. */
void strat_graph(Search *dist, const Workload *W, lbl_t *res, uint64_t *perr) {
    int p = dist->part.p, rank = dist->part.me;
    uint32_t *qidx = malloc((size_t)(W->q ? W->q : 1) * 4), nloc = 0;
    for (uint32_t j = 0; j < W->q; j++)
        if (part_owner(&dist->part, W->queries[j]) == rank) qidx[nloc++] = j;
    uint32_t *ids = malloc((size_t)W->k * 4);
    lbl_t *mine = malloc((size_t)(nloc ? nloc : 1) * W->B * sizeof(lbl_t));
    for (uint32_t s = 0; s < W->B; s++) {
        scenario_ids(W, s, ids);
        search_run(dist, W, ids, W->k);
        for (uint32_t j = 0; j < nloc; j++) mine[(size_t)s * nloc + j] = dist->key[W->queries[qidx[j]]];
    }
    int counts[MAXP], displs[MAXP], lcounts[MAXP], ldispls[MAXP], my_count = (int)nloc;
    uint32_t *all_idx = NULL;
    lbl_t *all_lbl = NULL;
    double t0 = MPI_Wtime();
    MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int r = 0; r < p; r++) lcounts[r] = counts[r] * (int)W->B;
        gather_displs(counts, displs, p);
        gather_displs(lcounts, ldispls, p);
        all_idx = malloc((size_t)(displs[p - 1] + counts[p - 1] + 1) * 4);
        all_lbl = malloc((size_t)(ldispls[p - 1] + lcounts[p - 1] + 1) * sizeof(lbl_t));
    }
    MPI_Gatherv(qidx, my_count, MPI_UINT32_T, all_idx, counts, displs, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Gatherv(mine, my_count * (int)W->B, MPI_UINT64_T, all_lbl, lcounts, ldispls, MPI_UINT64_T, 0,
                MPI_COMM_WORLD);
    timed_xfer(dist, t0);
    dist->st.msgs += (uint64_t)my_count * W->B;
    if (rank == 0) {
        size_t total = (size_t)W->B * W->q;
        uint8_t *placed = calloc(total, 1);
        for (int r = 0; r < p; r++)
            for (uint32_t s = 0; s < W->B; s++)
                for (int j = 0; j < counts[r]; j++) {
                    uint32_t qi = all_idx[displs[r] + j];
                    if (qi >= W->q) { (*perr)++; continue; }
                    res[(size_t)s * W->q + qi] = all_lbl[ldispls[r] + (size_t)s * counts[r] + j];
                    placed[(size_t)s * W->q + qi]++;
                }
        *perr += count_unplaced(placed, total);
        free(placed); free(all_idx); free(all_lbl);
    }
    free(qidx); free(ids); free(mine);
}
