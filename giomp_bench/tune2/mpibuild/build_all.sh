#!/bin/bash
# Build every compute-only MPI app at its default choices, to check the
# build scripts and the plugin's kernel reports before the tuner runs.
A=/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/apps
D=/shared/data1/Users/l1065028/giomp_bench/tune2/mpibuild
for a in himeno3_mpi jacobi_mpi miniweather_mpi moe_mpi cloverleaf_mpi minimod3_1000_mpi; do
  ( eval "$(python3 -c "import json;d=json.load(open('$A/$a/app.json'));print(' '.join(f'export {k}=\"{v}\";' for k,v in d['build']['env'].items()))")"
    GICC_SITES_OUT=$D/$a.sites.jsonl bash $A/${a%%_*}_dummy 2>/dev/null
    s=$(python3 -c "import json;print(json.load(open('$A/$a/app.json'))['build']['script'])")
    GICC_SITES_OUT=$D/$a.sites.jsonl bash /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/$s $D/$a > $D/$a.log 2>&1
    echo "$a: $([ -x $D/$a ] && echo BUILT || echo FAILED) kernels=$(grep -c dist_chunk $D/$a.sites.jsonl 2>/dev/null)" ) &
done; wait
