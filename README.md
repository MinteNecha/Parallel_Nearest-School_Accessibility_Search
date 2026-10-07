# Nearest-school road distance search with MPI (IT18X97 research project)

Compares three MPI work-distribution strategies for multi-source nearest-school search on a
Gauteng OpenStreetMap road graph, with weighted Dijkstra and unweighted BFS reported separately.

| Strategy | What is divided | Communication |
|---|---|---|
| `source` | schools of each scenario, dealt round-robin | one `MPI_Reduce(MPI_MIN)` over the q query entries |
| `graph` | the graph, in contiguous Morton-ordered ranges with equal edge counts | `MPI_Alltoallv` per superstep, `MPI_Gatherv` at the end |
| `scenario` | whole scenarios (independent school sets) | one `MPI_Gatherv` at the end |
| `seq` | nothing (baseline) | none |

The sequential baseline is one multi-source search per scenario (all schools start at distance 0).
Every result carries the nearest school's pool index and the distance, packed in one 64-bit label
(`distance << 32 | school_index`), so ties go to the lowest school index in all strategies.

## Requirements
Linux or WSL2, a C compiler, MPI (OpenMPI or MPICH), Python 3.10+ with numpy, pandas, scipy,
matplotlib and osmium (pyosmium).

    sudo apt install build-essential openmpi-bin libopenmpi-dev python3-pip
    pip install numpy pandas scipy matplotlib osmium

## Build and test
    make            # builds ./nsearch with mpicc -O2
    make test       # parallel results vs sequential, edge cases, and an end-to-end check vs SciPy

`make test` runs oversubscribed ranks for correctness only. Do not time anything that way.

## Data
Download two inputs yourself and record exactly what you used (file name, date, version, URL):
1. An OpenStreetMap extract covering Gauteng (`.osm.pbf`), for example from Geofabrik.
2. A public school list with a school identifier, latitude and longitude (CSV).

The school dataset name and version go into the paper's reference list. Do not cite a homepage.

Prepare three graph sizes (adjust the column names to your CSV; `--centre` is lat,lon):

    python3 tools/prepare_data.py --pbf gauteng.osm.pbf --schools schools.csv \
        --school-id-col EMIS --school-lat-col GIS_Latitude --school-lon-col GIS_Longitude \
        --centre=-26.2041,28.0473 --target-nodes 5000   --out data/small
    python3 tools/prepare_data.py ... --target-nodes 50000  --out data/medium
    python3 tools/prepare_data.py ... --target-nodes 500000 --out data/large

If your extract is larger than Gauteng, add `--bbox minlon,minlat,maxlon,maxlat` to clip it.
Check the printed `meta.json` for the real node counts: a crop cannot exceed what the extract holds,
and the largest strongly connected component is kept, so sizes are close to but not exactly the targets.
Highway classes default to primary, secondary, tertiary, unclassified, residential (and links);
change with `--highways`. One-way streets are respected unless `--ignore-oneway` is given.

Each output folder holds `graph.bin` (transposed CSR), `schools.bin`, `schools_map.csv`
(pool index to school id), `nodes.csv` and `meta.json` (counts, settings, SHA-256 of the inputs).

## Run
Quick check (small graph only, 3 trials, under 5 minutes):

    python3 tools/run_experiments.py --config quick --data data --mpi-args "--bind-to core"

Full matrix used for the paper (three graphs plus sweeps over k, q and B, 1 warm-up + 5 trials):

    python3 tools/run_experiments.py --config full --data data --mpi-args "--bind-to core"
    python3 tools/analyze.py
    make paper-assets

Use `--max-p` for the number of physical cores (default 4). The driver resumes where it stopped
if interrupted. `--synthetic` swaps in generated grid graphs for development only; never report
those numbers as Gauteng results.

Outputs in `results/`: `raw.csv` (every trial), `summary.csv`, `table_main.csv`,
`diff_dijkstra_vs_bfs.csv`, `figures/*.pdf`, `tables/*.tex`, `dumps/` (per-origin answers),
`environment.txt` (versions) and `design.json` (parameters of each cell).

## Seeds and parameters
Seed 42 everywhere. Query origins are sampled once from the node set. Scenario `s` is a random
k-subset of the school pool seeded from the seed and `s`. Defaults: k = pool/4, q = min(5000, n/5),
B = 8 (quick: q <= 1000, B = 4).

## Layout
    src/        C and MPI implementation
    tools/      data preparation, experiment driver, analysis and figures
    tests/      functional tests
    paper/      LNCS source (copy llncs.cls and splncs04.bst from Springer or Overleaf here)

## Known limits
Each rank holds the full read-only graph, so memory does not fall with p. Speedups from the graph
strategy depend on the edge cut of the Morton partition. Results from a laptop under WSL2 carry
scheduling noise; that is why medians and ranges are reported.
