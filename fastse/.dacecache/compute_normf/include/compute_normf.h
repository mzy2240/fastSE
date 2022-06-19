#include <dace/dace.h>
typedef void * compute_normfHandle_t;
extern "C" compute_normfHandle_t __dace_init_compute_normf(int Na, int Nb, int Nc);
extern "C" void __dace_exit_compute_normf(compute_normfHandle_t handle);
extern "C" void __program_compute_normf(compute_normfHandle_t handle, int * __restrict__ __return, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, int Na, int Nb, int Nc);
