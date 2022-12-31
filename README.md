The file `final.cu` implements in CUDA a variation on the SYMGS algorithm first implemented by professor Alberto Zeni. His implementation can still be found in `final.cu` in the CPU part, or in `/symgs/symgs-csr.c`.

How to run:
 - compile with: ` nvcc -o output final.cu -O3 `
 - execute with: `./output /directory/to/matrixfile/matrix.mtx ` 

On my architecture, the GPU implementation proved to be anywhere near 30% to 50% faster:
```
SYMGS Time CPU: 0.6764228344
SYMGS Time GPU: 0.3935761452
```

Refer to file `/report/report.pdf` for a better explanation and analysis of my implementation