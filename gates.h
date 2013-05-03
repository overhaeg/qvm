#include "cfloat.h"
#include "qureg_dense.h"
#include "config.h"
#include "string.h"

#ifndef GATES
#define GATES
typedef unsigned int uint;

uint permute(const uint i, const uint pos, const uint size) ;
uint alt_permute(const uint i, const uint pos, const uint size);
uint alt2_permute(const uint i, const uint npos, const uint nsize);

void base_quantum_X(const quantum_reg * const input, quantum_reg *output);
void general_quantum_X(const quantum_reg * const input, quantum_reg *output, uint qubit_position);
void general_quantum_Z(const quantum_reg * const input, quantum_reg *output, uint qubit_position);
void general_quantum_CZ(int tar1, int tar2, quantum_reg * qureg);

#endif
