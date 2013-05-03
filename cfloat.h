#include <CL/cl.h>
#include "math.h"

typedef cl_float2 COMPLEX_FLOAT;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef I
#define I ((COMPLEX_FLOAT)(0.0, 1.0))
#endif
#ifndef CFLOAT
#define CFLOAT
cl_float real(COMPLEX_FLOAT a);
cl_float imag(COMPLEX_FLOAT a);

cl_float cmod(COMPLEX_FLOAT a);
cl_float cangle(COMPLEX_FLOAT a);

COMPLEX_FLOAT cadd(COMPLEX_FLOAT a, COMPLEX_FLOAT b);
COMPLEX_FLOAT caddfl(COMPLEX_FLOAT a, cl_float b);
COMPLEX_FLOAT csub(COMPLEX_FLOAT a, COMPLEX_FLOAT b);
COMPLEX_FLOAT cmult(COMPLEX_FLOAT a, COMPLEX_FLOAT b);
COMPLEX_FLOAT cdiv(COMPLEX_FLOAT a, COMPLEX_FLOAT b);

COMPLEX_FLOAT csquare(COMPLEX_FLOAT a);

static inline cl_float 
quantum_prob_inline(COMPLEX_FLOAT a)
{
  cl_float r, i;

  r = real(a);
  i = imag(a);

  return r * r + i * i;
}


cl_float quantum_prob(COMPLEX_FLOAT a); 
COMPLEX_FLOAT quantum_conj(COMPLEX_FLOAT a);
COMPLEX_FLOAT quantum_cexp(cl_float phi);
#endif
