/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct compute_normf_2_t {

};

void __program_compute_normf_2_internal(compute_normf_2_t *__state, double * __restrict__ F, double * __restrict__ __return, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, int Na, int Nb, int Nc)
{

    {


        dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
        a, F, Na, 1);

        dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
        F, __return, ((Na + Nb) + Nc), 1);

    }
}

DACE_EXPORTED void __program_compute_normf_2(compute_normf_2_t *__state, double * __restrict__ F, double * __restrict__ __return, double * __restrict__ a, double * __restrict__ b, double * __restrict__ c, int Na, int Nb, int Nc)
{
    __program_compute_normf_2_internal(__state, F, __return, a, b, c, Na, Nb, Nc);
}

DACE_EXPORTED compute_normf_2_t *__dace_init_compute_normf_2(int Na, int Nb, int Nc)
{
    int __result = 0;
    compute_normf_2_t *__state = new compute_normf_2_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_compute_normf_2(compute_normf_2_t *__state)
{
    delete __state;
}

