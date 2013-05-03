#pragma OPENCL EXTENSION cl_khr_fp64 : enable 

typedef float2 COMPLEX_FLOAT;

inline float real(COMPLEX_FLOAT a){
     return a.x;
}
inline float imag(COMPLEX_FLOAT a){
     return a.y;
}

inline bool cequal(COMPLEX_FLOAT a, COMPLEX_FLOAT b)
{
    return a.x == b.x && a.y && b.y;
}

inline float cmod(COMPLEX_FLOAT a){
    return sqrt(real(a)*real(a) + imag(a)*imag(a));
}

inline float carg(COMPLEX_FLOAT a){

    if(a.x > 0){
        return atan(a.y / a.x);

    }else if(a.x < 0 && a.y >= 0){
        return atan(a.y / a.x) + M_PI;

    }else if(a.x < 0 && a.y < 0){
        return atan(a.y / a.x) - M_PI;

    }else if(a.x == 0 && a.y > 0){
        return M_PI/2;

    }else if(a.x == 0 && a.y < 0){
        return -M_PI/2;

    }else{
        return 0;
    }
}

inline COMPLEX_FLOAT cadd(COMPLEX_FLOAT a, COMPLEX_FLOAT b) {
    COMPLEX_FLOAT ret = {real(a)+real(b), imag(a) + imag(b)};
    return ret;
}

inline COMPLEX_FLOAT cmult(COMPLEX_FLOAT a, COMPLEX_FLOAT b){
    COMPLEX_FLOAT ret = { a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return ret;
}


inline COMPLEX_FLOAT cdiv(COMPLEX_FLOAT a, COMPLEX_FLOAT b){
    COMPLEX_FLOAT ret = {(a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y)};
    return ret;
}

inline COMPLEX_FLOAT csqrt(COMPLEX_FLOAT a){
    COMPLEX_FLOAT ret = { sqrt(cmod(a)) * cos(carg(a)/2),  sqrt(cmod(a)) * sin(carg(a)/2)};
    return ret;
 }

inline float quantum_prob(COMPLEX_FLOAT a)
{
  float r, i;

  r = real(a);
  i = imag(a);

  return r * r + i * i;
}

inline COMPLEX_FLOAT quantum_conj(COMPLEX_FLOAT a)
{
  float r, i;
  COMPLEX_FLOAT ret = {real(a), -imag(a)};

  return ret;
}

inline COMPLEX_FLOAT quantum_cexp(double phi)
{
    COMPLEX_FLOAT ret = {cos(phi), sin(phi)};
    return ret;
}

__kernel void kronecker(__global COMPLEX_FLOAT * result,
                        __global COMPLEX_FLOAT * input1,
                        __global COMPLEX_FLOAT * input2)
                        //__global int * tests)
{

    int gl_id = get_global_id(0);
    int loc_id = get_local_id(0);
	int gr_id = get_group_id(0);	
	int localSize = get_local_size(0);

    
    result[gl_id]=cmult(input1[loc_id],input2[gr_id]);


}


uint permute(const uint i, const uint pos, const uint size) {
  const uint rest = size / pos;
  return ( i % pos ) * rest + i / pos; 
}


__kernel void quantum_X(__global COMPLEX_FLOAT * input,
                        __global uint * target,
                        __local COMPLEX_FLOAT * input_permut,
                        __local COMPLEX_FLOAT * output_permut
                        )
{
 

 uint p_i;
 int gl_id = get_global_id(0);
 int globalSize = get_global_size(0);

 p_i = permute(gl_id, target[0], globalSize);
 input_permut[p_i] = input[gl_id];
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if ((gl_id % 2) == 0)
     output_permut[gl_id] = input_permut[gl_id+1];
 else output_permut[gl_id] = input_permut[gl_id-1];
 barrier(CLK_LOCAL_MEM_FENCE);

 p_i = permute(gl_id, (globalSize / target[0]), globalSize);
 input[p_i] = output_permut[gl_id];
}

__kernel void quantum_Z(__global COMPLEX_FLOAT * input,
                        __global int * target)
{
   int gl_id = get_global_id(0);
   int size = get_global_size(0);
   COMPLEX_FLOAT mult = {-1,0};
   int p_i = permute(gl_id, target[0], size);
   if (p_i % 2 == 1) 
    input[gl_id] = cmult(input[gl_id], mult);

}

__kernel void quantum_CZ(__global COMPLEX_FLOAT * input,
                         __global int * target)
{
    int global_id = get_global_id(0);
    COMPLEX_FLOAT mult = {-1,0};
    if ((global_id & target[0]) == target[0])
    {
        input[global_id] = cmult(input[global_id], mult);
    }


}

__kernel void quantum_diag_measure(__global COMPLEX_FLOAT * input,
                                   __global COMPLEX_FLOAT * output,
                                   __global uint * args,
                                   __global double * angle)
{

 int global_id = get_global_id(0);

 uint pos = args[0];
 uint size = args[1];
 uint upper_mask = args[2];
 uint lower_mask = args[3];
 COMPLEX_FLOAT cexp = quantum_cexp(-(angle[0]));

 double limit = (1.0/size) / 1000000;
 float prob;
 COMPLEX_FLOAT amp = {0,0} ;

 //assert( upper_mask + lower_mask == -1 );
 
 uint lpart = upper_mask & global_id<<1;
 uint rpart = lower_mask & global_id;

 uint k = lpart + rpart;
 uint i = k^pos;
 uint k_is_odd = k & pos; 

 COMPLEX_FLOAT zero = {0,0};

 if( !cequal(input[k],zero))
      amp = k_is_odd ? cadd(amp, -(cmult(input[k], cexp)))
	              : cadd(amp, input[k]);
 if( !cequal(input[i],zero))
      amp = k_is_odd ? cadd(amp, input[i]) 
	              : cadd(amp, -(cmult(input[i],cexp)));
 prob = quantum_prob( amp );
 if( prob > limit ) {
   // COMPLEX_FLOAT test = {k_is_odd, limit};
	output[global_id] = amp;
      }
      else {
             //COMPLEX_FLOAT test = {k_is_odd, rpart};
             output[global_id] = zero;
      }

}
