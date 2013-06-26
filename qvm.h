#ifndef QVM_H
#define QVM_H

//#include "libquantum/complex.h"
//#include "libquantum/error.h"
//#include <quantum.h>

//#include "cfloat.h"
#include "qureg_dense.h"
#include "gates.h"
#include <CL/cl.h>
#include <papi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_UNSIGNED unsigned long long

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context;
cl_command_queue command_queue;
cl_int ret;
cl_program program;
size_t max_group;
long_long start, end;

extern void quantum_copy_qureg(quantum_reg *src, quantum_reg *dst);
extern void quantum_delete_qureg_hashpreserve(quantum_reg *reg);

typedef int qid_t;
typedef int tangle_size_t;
typedef int pos_t;

#endif
