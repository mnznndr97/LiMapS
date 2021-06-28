#pragma once

#include <cublas.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                     \
{                                                                              \
    const cublasStatus_t err = call;                                           \
    if (err != CUBLAS_STATUS_SUCCESS)                                          \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
         assert(false);                                                        \
    }                                                                          \
}