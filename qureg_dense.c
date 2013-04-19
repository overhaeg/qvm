#include "qureg_dense.h"

typedef unsigned int uint;

unsigned long quantum_memman(long change)
{
  static long mem = 0, max = 0;

  mem += change;

  if(mem > max)
    max = mem;

  return mem;
}

quantum_reg 
quantum_new_qureg ( int size)
{
    quantum_reg reg;
    
    /*Allocate memory*/
    int i;
    COMPLEX_FLOAT one = 1;
    reg.amplitudes = malloc(size * sizeof(COMPLEX_FLOAT));
    for (i = 0; i<size; i++)
	    reg.amplitudes[i] = one;
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
    quantum_memman(-reg->size*sizeof(COMPLEX_FLOAT));
    reg->amplitudes = 0;
}

void
quantum_copy_qureg(quantum_reg* src, quantum_reg* dst)
{
   *dst = *src;
   dst->amplitudes =  calloc(dst->size, sizeof(COMPLEX_FLOAT));
   quantum_memman(dst->size*sizeof(COMPLEX_FLOAT));
  // memcpy(dst->amplitudes, src->amplitudes, src->size*sizeof(uint));
}

quantum_reg
quantum_kronecker (quantum_reg *reg1, quantum_reg *reg2)
{
   int i,j;
   quantum_reg reg;

   reg.size = reg1->size + reg2->size;
   reg.amplitudes = calloc(reg.size, sizeof(COMPLEX_FLOAT));
   
   for(i=0; i<reg1->size;i++)
    for(j=0; j<reg2->size;j++)
        reg.amplitudes[i*reg2->size+j]=reg1->amplitudes[i]*reg2->amplitudes[j];
   
   return reg;

}


uint Log2( uint x )
{
  uint ans = 0 ;
  while( x>>=1 ) ans++;
  return ans ;
}

void
quantum_print_qureg(quantum_reg reg)
{
  int i,j;
  
  for(i=0; i<reg.size; i++)
    {
      printf("% f %+fi|%u> (%e) (|", quantum_real(reg.amplitudes[i]),
	     quantum_imag(reg.amplitudes[i]), i, 
	     quantum_prob_inline(reg.amplitudes[i]));
      for(j=Log2(reg.size)-1;j>=0;j--)
	{
	  if(j % 4 == 3)
	    printf(" ");
	  printf("%i", (((1 << j) & i) > 0));
	}

      printf(">)\n");
    }

  printf("\n");
}
    
