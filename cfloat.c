//Source: http://pastebin.com/2XUUTTqa & http://stackoverflow.com/questions/10016125/complex-number-support-in-opencl
//#include "config.h"
#include <CL/cl.h>
#include "math.h"
#include "cfloat.h"

/*
 * Return Real (Imaginary) component of complex number:
 */
cl_float real(COMPLEX_FLOAT a){
     return a.s[0];
}
cl_float imag(COMPLEX_FLOAT a){
     return a.s[1];
}

/*
 * Get the modulus of a complex number (its length):
 */
cl_float 
cmod(COMPLEX_FLOAT a){
    return sqrt(real(a)*real(a) + imag(a)*imag(a));
}

/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
cl_float 
cangle(COMPLEX_FLOAT a){

    if(real(a) > 0){
        return atan(imag(a) / real(a));

    }else if(real(a) < 0 && imag(a) >= 0){
        return atan(imag(a) / real(a)) + M_PI;

    }else if(real(a) < 0 && imag(a) < 0){
        return atan(imag(a) / real(a)) - M_PI;

    }else if(real(a) == 0 && imag(a) > 0){
        return M_PI/2;

    }else if(real(a) == 0 && imag(a) < 0){
        return -M_PI/2;

    }else{
        return 0;
    }
}

COMPLEX_FLOAT
cadd(COMPLEX_FLOAT a, COMPLEX_FLOAT b) {
    COMPLEX_FLOAT ret = {real(a)+real(b), imag(a) + imag(b)};
    return ret;
}

COMPLEX_FLOAT
caddfl(COMPLEX_FLOAT a, cl_float b) {
    COMPLEX_FLOAT ret = {real(a) + b, imag(a)};
    return ret;
}


COMPLEX_FLOAT
csub(COMPLEX_FLOAT a, COMPLEX_FLOAT b) {

    COMPLEX_FLOAT ret = {real(a)-real(b), imag(a) - imag(b)};
    return ret;
}


/*
 * Multiply two complex numbers:
 *
 *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
 */
COMPLEX_FLOAT
cmult(COMPLEX_FLOAT a, COMPLEX_FLOAT b){
    COMPLEX_FLOAT ret = { real(a)*real(b) - imag(a)*imag(b), real(a)*imag(b) + imag(a)*real(b)};
    return ret;
}


/*
 * Divide two complex numbers:
 *
 *  aReal + I*aImag     (aReal + I*aImag) * (bReal - I*bImag)
 * ----------------- = ---------------------------------------
 *  bReal + I*bImag     (bReal + I*bImag) * (bReal - I*bImag)
 * 
 *        aReal*bReal - I*aReal*bImag + I*aImag*bReal - I^2*aImag*bImag
 *     = ---------------------------------------------------------------
 *            bReal^2 - I*bReal*bImag + I*bImag*bReal  -I^2*bImag^2
 * 
 *        aReal*bReal + aImag*bImag         aImag*bReal - Real*bImag 
 *     = ---------------------------- + I* --------------------------
 *            bReal^2 + bImag^2                bReal^2 + bImag^2
 * 
 */
COMPLEX_FLOAT 
cdiv(COMPLEX_FLOAT a, COMPLEX_FLOAT b){
    COMPLEX_FLOAT ret = {(real(a)*real(b) + imag(a)*imag(b))/(real(b)*real(b) + imag(b)*imag(b)), (imag(a)*real(b) - real(a)*imag(b))/(real(b)*real(b) + imag(b)*imag(b))};
    return ret;
}


/*
 *  Square root of complex number.
 *  Although a complex number has two square roots, numerically we will
 *  only determine one of them -the principal square root, see wikipedia
 *  for more info: 
 *  http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
COMPLEX_FLOAT csquare(COMPLEX_FLOAT a){
     COMPLEX_FLOAT ret = { sqrt(cmod(a)) * cos(cangle(a)/2),  sqrt(cmod(a)) * sin(cangle(a)/2)};
     return ret;
 }

cl_float
quantum_prob(COMPLEX_FLOAT a)
{
  return quantum_prob_inline(a);
}


COMPLEX_FLOAT
quantum_conj(COMPLEX_FLOAT a)
{
  float r, i;
  COMPLEX_FLOAT ret = {real(a), -imag(a)};

  return ret;
}


/* Calculate e^(i * phi) */

COMPLEX_FLOAT 
quantum_cexp(cl_float phi)
{
    COMPLEX_FLOAT ret = {cos(phi), sin(phi)};
    return ret;
}



