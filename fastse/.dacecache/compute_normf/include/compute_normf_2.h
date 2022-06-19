#include <dace/dace.h>
typedef void * compute_normf_2Handle_t;
extern "C" compute_normf_2Handle_t __dace_init_compute_normf_2(int Na, int Nb, int Nc);
extern "C" void __dace_exit_compute_normf_2(compute_normf_2Handle_t handle);
extern "C" void __program_compute_normf_2(compute_normf_2Handle_t handle, double * __restrict__ F, double * __restrict__ __return, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, int Na, int Nb, int Nc);
