#include "config.h"
#include <stdio.h>
#include "stdlib.h"
#include "complex.h"
#include "math.h"

typedef struct qu_reg_struct
{
  COMPLEX_FLOAT* amplitudes; 
  size_t size;      
} quantum_reg;

quantum_reg quantum_new_qureg(int size);
void quantum_delete_qureg(quantum_reg *reg);
void quantum_copy_qureg(quantum_reg *src, quantum_reg *dst);
quantum_reg quantum_kronecker(quantum_reg *reg1, quantum_reg *reg2);
void quantum_print_qureg(quantum_reg reg);






