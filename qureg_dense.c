#include "qureg_dense.h"
    
quantum_reg 
quantum_new_qureg (MAX_UNSIGNED initval, int width)
{
    quantum_reg reg;
    
    /*Allocate memory*/

    reg.amplitudes = calloc(width, sizeof(uint));
    reg.size = width;
    /*Mem manager?*/

    return reg;

}


/*quantum_reg
quantum_new_qureg_size Useful?
*/        

void
quantum_delete_qureg(quantum_reg *reg)
{
    free(reg->amplitudes);
    /*Mem manager?*?*/
    reg->amplitudes = 0;
}

quantum_reg
quantum_kronecker (quantum_reg *reg1, quantum_reg *reg2)
{
   quantum_reg reg;
   int i,j;
   reg.size = reg1->size + reg2->size;
   reg.amplitudes = calloc(reg.size, sizeof(uint));
   
   for(i=0; i<reg1->size;i++)
    for(j=0; j<reg2->size;j++)
        reg.amplitudes[i*reg2->size+j]=reg1->amplitudes[i]*reg2->amplitudes[j];
   
   return reg;

}

