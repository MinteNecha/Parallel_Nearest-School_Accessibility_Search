CC = mpicc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
SRC = src/main.c src/graph.c src/workload.c src/search.c src/strategies.c

nsearch: $(SRC) src/nsearch.h
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

clean:
	rm -f nsearch

.PHONY: clean

test: nsearch
	bash tests/run_tests.sh

paper-assets:
	mkdir -p paper/figures paper/tables
	cp results/figures/*.pdf paper/figures/
	cp results/tables/*.tex paper/tables/

.PHONY: test paper-assets
