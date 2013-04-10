#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "qureg_dense.h"
#include "config.h"

// Arguments:
//  -    i: any integer 0 <= i < size
//  - pos : a power of 2 integer (2^k), 0 <= m < nlog2(size)
//  - size: a power of 2 integer (2^n)
// Details:
//   Permute will rotate the binary numeral representation of 'i' a log2(pos)
//   number of times to the right. The bit masked by pos = 2^k will after 
//   rotation be at position 0 (last position).
// Invariants:
//   permute(i,1,n) == permute(i,n,n) == i
//   permute( permute(i,m,n), n/m, n ) == i  (for any valid m)
// Example: i=3 (0b0011), pos=2 (0b0010), size=16
//      (0011 --> 1001)    (0001 --> 1000)
//          3 --> 9            1 --> 8
// Example: i=4 (0b0100), pos=4 (0b0100), size=16
//      (0100 --> 0001)
//          4 --> 1
uint permute(const uint i, const uint pos, const uint size) {
  const uint rest = size / pos;
  return ( i % pos ) * rest + i / pos; 
}

// Alternative implementation, using bit-level operations.
// In a lot of cases, you don't actually need to do the permutation, but
//  just do some bit fuckery.
uint alt_permute(const uint i, const uint pos, const uint size) {
  const uint lower_mask = pos-1;
  const uint upper_mask = ~lower_mask;
  const uint rest = size / pos;
  return ((i & lower_mask) * rest) ^ ((i & upper_mask) / pos);
}

// Another alternative way to do bitfuckery, this time passing the 
// position as a bit index (0 <= npos < size)
uint alt2_permute(const uint i, const uint npos, const uint nsize) {
  const uint lower_mask = (1<<npos) - 1;
  const uint upper_part = (i & lower_mask) << (nsize - npos);
  const uint lower_part = (i & ~lower_mask) >> npos;
  return upper_part ^ lower_part;
}



#define RUN_TEST( FUNCALL ) \
  memset(test_result, 0, sizeof(test_result));  \
  for( i=0; i<32; ++i ) {                       \
    p_i = FUNCALL;                     \
    test_result[p_i] = test_input[i];           \
  }                                             \
  printf("\t"#FUNCALL" results:\t { ");    \
  for( i=0; i<31; ++i ) {                       \
    printf( " %u,", test_result[i] );           \
  }                                             \
  printf( " %u }\n\n", test_result[31] )          \


void base_quantum_X(const quantum_reg * const input, quantum_reg *output) {
  int i;
  for( i=0; i<input->size; i+=2 ) {
    output->amplitudes[i+1] = input->amplitudes[i];
    output->amplitudes[i]   = input->amplitudes[i+1];
  }
}

// qubit_position here can be 1 (0b0001), 2 (0b0010), 4 (0b0100) or 8 (0b1000)
void general_quantum_X(const quantum_reg * const input, quantum_reg *output, uint qubit_position) {
  int i, p_i;
  const size_t size = input->size;
  quantum_reg permuted_input = quantum_new_qureg(1, size); 
  quantum_reg permuted_output = quantum_new_qureg(1, size);

  // permute the input amplitudes
  for( i=0; i<size; ++i ) {
    p_i = permute(i, qubit_position, size);
    permuted_input.amplitudes[p_i] = input->amplitudes[i];
  }

  base_quantum_X( &permuted_input, &permuted_output );

  // permute back to original order
  for( i=0; i<size; ++i ) {
    p_i = permute(i, (size / qubit_position), size);
    output->amplitudes[p_i] = permuted_output.amplitudes[i];
  }
}

void general_quantum_Z(const quantum_reg * const input, quantum_reg *output, uint qubit_position) {

output->amplitudes = input->amplitudes;


}
  
/*************************************************/

void print_array(const char* name, const uint * const array, size_t size) {
  int i;
  printf("\t%s:\t\t\t { ", name);
  for( i=0; i<size-1; ++i )
    printf("%u, ", array[i]);
  printf("%u }\n\n", array[size-1]);
}

int main( int argc, char* argv[] ) {
  uint i, p_i;
  uint test_input[32]  = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  uint test_result[32];

  printf("\ntesting the permutation functions on vector {0,1,..,15}\n");
  RUN_TEST(      permute(i, 2, 32) );
  RUN_TEST(  alt_permute(i, 2, 32) );
  RUN_TEST( alt2_permute(i, 2,  4) );
  

  printf("\ntesting the quantum ops:\n");
  quantum_reg input = quantum_new_qureg(1,32); 
  quantum_reg output = quantum_new_qureg(1,32);
  input.amplitudes = test_input;

  base_quantum_X( &input, &output);
  print_array( "after basic X", output.amplitudes, 32 ); 

  general_quantum_X( &input, &output, 2);
  print_array( "after X_2", output.amplitudes, 32 );

  general_quantum_X( &input, &output, 4);
  print_array( "after X_4", output.amplitudes, 32 );

  general_quantum_X( &input, &output, 8);
  print_array( "after X_8", output.amplitudes, 32 );
    
  return 0;
}
