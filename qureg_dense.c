#include "qureg_dense.h"

unsigned long quantum_memman(long change)
{
  static long mem = 0, max = 0;

  mem += change;

  if(mem > max)
    max = mem;

  return mem;
}

quantum_reg 
quantum_new_qureg (MAX_UNSIGNED initval, int size)
{
    quantum_reg reg;
    
    /*Allocate memory*/

    reg.amplitudes = calloc(size, sizeof(COMPLEX_FLOAT));
    reg.size = size;
    quantum_memman(size * sizeof(COMPLEX_FLOAT));

    return reg;

}



/*quantum_reg
quantum_new_qureg_size Useful?
*/        

void
quantum_delete_qureg(quantum_reg *reg)
{
    free(reg->amplitudes);
    quantum_memman(-reg->size*sizeof(COMPLEX_FLOAT);
    reg->amplitudes = 0;
}

void
quantum_copy_qureg(quantum_reg* src, quantum_reg* dest)
{
   *dst = *src;
   dst->amplitudes =  calloc(dst->size, sizeof(COMPLEX_FLOAT));
   quantum_memman(size*sizeof(COMPLEX_FLOAT));
   memcpy(dst->amplitudes, src->amplitudes, src->size*sizeof(COMPLEX_FLOAT));
}

quantum_reg
quantum_kronecker (quantum_reg *reg1, quantum_reg *reg2)
{
   quantum_reg reg;

   reg.size = reg1->size + reg2->size;
   reg.amplitudes = calloc(reg.size, sizeof(COMPLEX_FLOAT);
   
   for(i=0; i<reg1->size;i++)
    for(j=0; j<reg2->size;j++)
        reg.amplitudes[i*reg2->size+j]=reg1.amplitudes[i]*reg2.amplitudes[j];
   
   return reg;

}

