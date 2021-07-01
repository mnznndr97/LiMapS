#pragma once

#include <cublas_v2.h>

#if _DEBUG
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
#else
#define CUBLAS_CHECK(call)                                                     \
{                                                                              \
    const cublasStatus_t err = call;                                           \
    if (err != CUBLAS_STATUS_SUCCESS)                                          \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(-1);                                                              \
    }                                                                          \
}
#endif