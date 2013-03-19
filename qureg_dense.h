struct qu_reg
{
  COMPLEX_FLOAT* amplitudes; 
  size_t size;      
} quantum_reg;

quantum_reg quantum_new_reg(MAX_UNSIGNED initval, int size);
void quantum_delete_qureg(quantum_reg *reg);

quantum_reg quantum_kronecker(quantum_reg *reg1, quantum_reg *reg2);






