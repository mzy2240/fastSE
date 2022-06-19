#include <cstdlib>
#include "../include/compute_normf.h"

int main(int argc, char **argv) {
    compute_normfHandle_t handle;
    int Na = 42;
    int Nb = 42;
    int Nc = 42;
    int * __restrict__ __return = (int*) calloc(1, sizeof(int));
    double * __restrict__ a = (double*) calloc(Na, sizeof(double));
    double * __restrict__ b = (double*) calloc(Nb, sizeof(double));
    double * __restrict__ c = (double*) calloc(Nc, sizeof(double));


    handle = __dace_init_compute_normf(Na, Nb, Nc);
    __program_compute_normf(handle, __return, a, b, c, Na, Nb, Nc);
    __dace_exit_compute_normf(handle);

    free(__return);
    free(a);
    free(b);
    free(c);


    return 0;
}
