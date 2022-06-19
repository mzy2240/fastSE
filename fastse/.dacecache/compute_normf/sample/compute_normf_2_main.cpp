#include <cstdlib>
#include "../include/compute_normf_2.h"

int main(int argc, char **argv) {
    compute_normf_2Handle_t handle;
    int Na = 42;
    int Nb = 42;
    int Nc = 42;
    double * __restrict__ F = (double*) calloc(((Na + Nb) + Nc), sizeof(double));
    double * __restrict__ __return = (double*) calloc(((Na + Nb) + Nc), sizeof(double));
    double * __restrict__ a = (double*) calloc(Na, sizeof(double));
    double * __restrict__ b = (double*) calloc(Nb, sizeof(double));
    double * __restrict__ c = (double*) calloc(Nc, sizeof(double));


    handle = __dace_init_compute_normf_2(Na, Nb, Nc);
    __program_compute_normf_2(handle, F, __return, a, b, c, Na, Nb, Nc);
    __dace_exit_compute_normf_2(handle);

    free(F);
    free(__return);
    free(a);
    free(b);
    free(c);


    return 0;
}
