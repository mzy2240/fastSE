/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct compute_normf_0_t {

};

void __program_compute_normf_0_internal(compute_normf_0_t *__state, int * __restrict__ __return, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, int Na, int Nb, int Nc)
{

    {

        {
            int __out;

            ///////////////////
            // Tasklet code (assign_18_4)
            __out = ((Na + Nb) + Nc);
            ///////////////////

            __return[0] = __out;
        }

    }
}

DACE_EXPORTED void __program_compute_normf_0(compute_normf_0_t *__state, int * __restrict__ __return, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, int Na, int Nb, int Nc)
{
    __program_compute_normf_0_internal(__state, __return, a, b, c, Na, Nb, Nc);
}

DACE_EXPORTED compute_normf_0_t *__dace_init_compute_normf_0(int Na, int Nb, int Nc)
{
    int __result = 0;
    compute_normf_0_t *__state = new compute_normf_0_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_compute_normf_0(compute_normf_0_t *__state)
{
    delete __state;
}

