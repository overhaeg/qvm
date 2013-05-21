#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>

#include <sexp.h>
#include <sexp_ops.h>
#include <sexp_vis.h>
//#include <quantum.h>


//#include "libquantum/config.h"

#include "bitmask.h"
#include "qvm.h"

#define STRING_SIZE (size_t)UCHAR_MAX	
#define MAX_TANGLES (size_t)SHRT_MAX
#define MAX_QUBITS  (size_t)SHRT_MAX
#define MAX_SOURCE_SIZE (0x100000)

#define car hd_sexp
#define cdr next_sexp
#define max(x,y) x < y ? x : y


int _verbose_ = 0;
int _alt_measure_ = 0;
quantum_reg _proto_diag_qubit_;
quantum_reg _proto_dual_diag_qubit_;
cl_mem target_buffer; //buffer of size one used to send arguments for gates X, Z, CZ


/************
 ** OPENCL **
 ************/


void CheckErr (cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		printf("ERROR: %s (%d)", name, err);
		exit(EXIT_FAILURE);
	}
}

void free_opencl() {

    ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseProgram(program);
    ret = clReleaseMemObject(target_buffer);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);


}

void init_opencl() {
   
 cl_uint ret_num_devices;
 cl_uint ret_num_platforms;
 ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
 CheckErr(ret, "PlatformIDs, line 50: ");
 ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
 CheckErr(ret, "DeviceIDs, line 52");
 
 context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 CheckErr(ret, "CreateContext, line 61: ");
 command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 CheckErr(ret, "CreateCommandQueue, line 63: ");

 clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_group), &max_group, NULL);

 
}

cl_program build_program(char *file){

    FILE *fp;
	char *source_str;
	size_t source_size;
	fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel, \n");
		exit(1);
		}

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	CheckErr(ret, "CreateProgramWithSource, line 65: ");
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
//	if (ret == CL_BUILD_PROGRAM_FAILURE) {
	    // Determine the size of the log
	    size_t log_size;
    	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	    // Allocate memory for the log
    	char *log = (char *) malloc(log_size);

    	// Get the log
    	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    	// Print the log
    	printf("%s\n", log);
//	}

    return program;
}

cl_mem qureg_to_buffer(quantum_reg reg) {

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, reg.size * sizeof(COMPLEX_FLOAT), NULL,&ret);
	CheckErr(ret, "CreateBuffer: ");
    ret = clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, 0, reg.size * sizeof(COMPLEX_FLOAT), reg.amplitudes, 0, NULL, NULL);
    CheckErr(ret, "EnqueueWriteBuffer: ");

    return buffer;
}

quantum_reg buffer_to_qureg(cl_mem buffer, int qubits) {

    quantum_reg newqureg = quantum_new_qureg(qubits);
    size_t size = pow(2,qubits);
    COMPLEX_FLOAT * amplitudes = malloc(size * sizeof(COMPLEX_FLOAT));
    ret = clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, size * sizeof(COMPLEX_FLOAT), amplitudes, 0 , NULL, NULL);
    CheckErr(ret, "ResultBuffer: ");
    newqureg.amplitudes = amplitudes;
    return newqureg;



}
/*
uint Log2( uint x )
{
  uint ans = 0 ;
  while( x>>=1 ) ans++;
  return ans ;
}
*/
cl_mem parralel_kronecker(cl_mem qureg1, cl_mem qureg2, size_t size1, size_t size2) {
    
    uint multiplier = 0;
 
    if(size2 > max_group)
    {
      multiplier = size2/max_group;
      size2 = max_group;
    }
      
    size_t newsize = size1 * size2;
    cl_mem newqureg = clCreateBuffer(context, CL_MEM_READ_WRITE, newsize * sizeof(COMPLEX_FLOAT), NULL, &ret);
    CheckErr(ret, "CreateBuffer: ");
    
    ret = clEnqueueWriteBuffer(command_queue, target_buffer, CL_TRUE, 0, sizeof(cl_uint), &multiplier, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "kronecker", &ret);
	CheckErr(ret, "CreateKernel, line 90: ");
   
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&newqureg);
	CheckErr(ret, "KroSetKernelArg0, line 93: ");
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&qureg1);
	CheckErr(ret, "KroSetKernelArg1, line 96: ");
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&qureg2);
	CheckErr(ret, "KroSetKernelArg2, line 99: ");
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&target_buffer);
    CheckErr(ret, "KroSetKernelArg3, line 99: ");

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &newsize, &size2, 0, NULL, NULL);
	CheckErr(ret, "Kro.EnqueueNDRangeKernel, line 106: ");
  /*
    int qubits = Log2(newsize);
    quantum_reg qureg = buffer_to_qureg(newqureg, qubits);
    printf("\n size1= %u, size2=%u, newsize=%u qubits=%u\n", size1, size2, newsize, qubits);
    quantum_print_qureg(qureg);
    */
    ret = clReleaseKernel(kernel);
    
    return newqureg;

}

void parralel_quantum_X(cl_mem input_buffer, size_t size, cl_uint qubit_position) {

  size_t groupSize = size>max_group ? max_group : size;
  cl_int target = 1 << qubit_position;

   
  ret = clEnqueueWriteBuffer(command_queue, target_buffer, CL_TRUE, 0, sizeof(cl_int), &target, 0, NULL, NULL);  
  CheckErr(ret, "WriteBuffer_X:" ); 

  cl_kernel kernel = clCreateKernel(program, "quantum_X", &ret);
  CheckErr(ret, "CreateKernel, line 90: ");
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_buffer);
  CheckErr(ret, "SetKernelArg, line 93: ");
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&target_buffer);
  CheckErr(ret, "SetKernelArg: ");
  ret = clSetKernelArg(kernel, 2, size * sizeof(COMPLEX_FLOAT), NULL);
  CheckErr(ret, "SetKernelArg, line 99: ");
  ret = clSetKernelArg(kernel, 3, size * sizeof(COMPLEX_FLOAT), NULL);
  CheckErr(ret, "SetKernelArg, line 99: ");

   

  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &size, &groupSize, 0, NULL, NULL);
  CheckErr(ret, "X.EnqueueNDRangeKernel, line 106: ");

  ret = clReleaseKernel(kernel);
}

void parralel_quantum_Z(cl_mem input_buffer, size_t size, int qubit_position) {
    
    size_t groupSize = size>max_group ? max_group : size;
    
    cl_int target = 1 << qubit_position;
    ret = clEnqueueWriteBuffer(command_queue, target_buffer, CL_TRUE, 0, sizeof(cl_int), &target, 0, NULL, NULL);
    CheckErr(ret, "WriteBuffer_Z");

    cl_kernel kernel = clCreateKernel(program, "quantum_Z", &ret);
    CheckErr(ret, "CreateKernel: ");
    ret = clSetKernelArg(kernel, 0 , sizeof(cl_mem), (void*)&input_buffer);
    CheckErr(ret, "SetKernelArg");
    ret = clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void*)&target_buffer);
    CheckErr(ret, "SetKernelArg");

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &size, &size, 0, NULL, NULL);
    CheckErr(ret, "Z.EnqueueNDRangeKernel: ");

    ret = clReleaseKernel(kernel);
}

void parralel_quantum_CZ(cl_mem input_buffer, size_t size, cl_int qid1, cl_int qid2) {

   size_t groupSize = size>max_group ? max_group : size;
    

    cl_int bitmask = (1 << qid1) | (1 << qid2);

    ret = clEnqueueWriteBuffer(command_queue, target_buffer, CL_TRUE, 0, sizeof(cl_int), &bitmask, 0, NULL, NULL);
    CheckErr(ret, "WriteBuffer_CZ");

    cl_kernel kernel = clCreateKernel(program, "quantum_CZ", &ret);
    CheckErr(ret, "CreateKernel: ");
    ret = clSetKernelArg(kernel, 0 , sizeof(cl_mem), (void*)&input_buffer);
    CheckErr(ret, "SetKernelArg");
    ret = clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void*)&target_buffer);
    CheckErr(ret, "SetKernelArg");

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &size, &groupSize, 0, NULL, NULL);
    CheckErr(ret, "CZ.EnqueueNDRangeKernel: ");


    ret = clReleaseKernel(kernel);
}

cl_mem parralel_diag_measure(cl_mem input_buffer, cl_int pos, cl_double angle, cl_int qubits)
{
  
  size_t size = 1 << (qubits-1);
  
  size_t groupSize = size>max_group ? max_group : size;
  cl_uint lpart, rpart, k_even, k_odd, k, i; 
  int global_id;

  cl_uint * args = malloc(2 * sizeof(cl_uint));
  args[0] = 1 << pos;
  args[1] = 1 << qubits;
  args[2] = ((uint)(-1/args[0]))*args[0];
  args[3] = -1 % args[0];
 // printf("\ntest²: %u, %u, %u, %u, %u %f\n", pos, args[0], args[1], args[2], args[3], angle);
 /*  
  for(global_id = 0; global_id < size; global_id++){

  lpart = (args[2] & global_id)<<1;
  rpart = args[3] & global_id;
  
  k_even = (lpart^rpart) & ~args[0];  //(~pos2 is 11…11011…11, k is dus 'even' per constructie)
  k_odd  = (lpart^rpart) + args[0];
  k = lpart+rpart;
  i = k^args[0];

  printf("\ntest³: %u, %u, %u, %u, %u, %u", lpart, rpart, k_even, k_odd, k, i);
          }

*/
  //quantum_reg qureg = buffer_to_qureg(input_buffer, qubits);
  //printf("\n size1= %u, size2=%u, newsize=%u qubits=%u\n", size1, size2, newsize, qubits);
  //quantum_print_qureg(qureg);

  cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(COMPLEX_FLOAT), NULL, &ret);
  CheckErr(ret, "diag_outputbuffer");
  cl_mem args_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(cl_uint), NULL, &ret);
  CheckErr(ret, "diag_argbuffer");
  cl_mem angle_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ret);
  CheckErr(ret, "M_anglebuffer");

  //cl_mem dump_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(cl_double), NULL, &ret);
  ret = clEnqueueWriteBuffer(command_queue, args_buffer, CL_TRUE, 0, 4* sizeof(cl_uint), args, 0, NULL, NULL);
  CheckErr(ret, "M.args.WriteBuffer");
  ret= clEnqueueWriteBuffer(command_queue, angle_buffer, CL_TRUE, 0, sizeof(cl_double), &angle, 0, NULL, NULL);
  CheckErr(ret, "M.angle.WriteBuffer");
 // cl_uint * vars = malloc(4 * sizeof(cl_uint));
 // cl_double * testangle = malloc(sizeof(cl_double));
 // ret = clEnqueueReadBuffer(command_queue, args_buffer, CL_TRUE, 0, 4 * sizeof(cl_uint), vars, 0 , NULL, NULL);
 // ret = clEnqueueReadBuffer(command_queue, angle_buffer, CL_TRUE, 0, sizeof(cl_double), testangle, 0, NULL, NULL);
  //printf("\ntest⁴: %u %u %u %u %f \n", vars[0], vars[1], vars[2], vars[3], testangle[0]);
 // ret = clEnqueueWriteBuffer(command_queue, angle_buffer, CL_TRUE, 0, sizeof(cl_double), &angle, 0, NULL, NULL);
 // CheckErr(ret, "x4");

  cl_kernel kernel = clCreateKernel(program, "quantum_diag_measure", &ret);
  CheckErr(ret, "M.CreateKernel");
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_buffer);
  CheckErr(ret, "M.SetArg0");
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_buffer);
  CheckErr(ret, "M.SetArg1");
  //ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&angle_buffer);
  //CheckErr(ret, "x3");
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&args_buffer);
   CheckErr(ret, "M.SetArg2");
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&angle_buffer);
  CheckErr(ret, "m.SetArg3");
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1 , NULL, &size, &groupSize, 0, NULL, NULL);
  CheckErr(ret, "M.EnqueueNDRangeKernel");
  //ret = clEnqueueReadBuffer(command_queue, args_buffer, CL_TRUE, 0, 3 * sizeof(cl_double), args, 0 , NULL, NULL);
  //printf("\ntest⁴: %f %f %f\n", args[0], args[1], args[2]);
  
  //qureg = buffer_to_qureg(input_buffer, qubits);
  //printf("\n size1= %u, size2=%u, newsize=%u qubits=%u\n", size1, size2, newsize, qubits);
  //quantum_print_qureg(qureg);

  //qureg = buffer_to_qureg(output_buffer, qubits - 1 );
  //quantum_print_qureg(qureg);

  ret = clReleaseKernel(kernel);
  CheckErr(ret, "mReleaseKernel");
  ret = clReleaseMemObject(input_buffer);
  CheckErr(ret, "mReleaseinput");
  ret = clReleaseMemObject(args_buffer);
  CheckErr(ret, "mReleaseargs");
  ret= clReleaseMemObject(angle_buffer);
  CheckErr(ret, "mReleaseangle");
  return output_buffer; 


}


/************
 ** TANGLE **
 ************/
typedef struct qid_list {
  qid_t qid;
  struct qid_list* rest;
} qid_list_t;

typedef struct tangle { 
  tangle_size_t size;
  qid_list_t* qids;
  quantum_reg qureg;
  cl_mem qureg_buffer;
 } tangle_t;  

tangle_t* init_tangle() {
  tangle_t* tangle = (tangle_t*) malloc(sizeof(tangle_t));   //ALLOC tangle
  tangle->size = 0;
  tangle->qids = NULL;
  return tangle;
}

void free_qid_list( qid_list_t* qids) {
  qid_list_t* rest = NULL;
  while( qids ) {
    rest = qids->rest;
    free( qids ); 
    qids = rest;
  }
}

void free_tangle( tangle_t* tangle ) {
  free_qid_list( tangle->qids );
  tangle->size = 0;
  tangle->qids = NULL;
  quantum_delete_qureg( &tangle->qureg );
  cl_mem test = tangle->qureg_buffer;
  clReleaseMemObject(test);
  free( tangle ); //FREE tangle
}

void print_qids( const qid_list_t* qids ) {
  printf("[");
const qid_list_t* cons = qids;
  for( ;
       cons;
       cons=cons->rest ) {
    printf("%d", cons->qid);
    if( cons->rest )
      printf(", ");
  }
  printf("]");
}

void print_tangle( const tangle_t* restrict tangle ) {
  assert( tangle );
  print_qids( tangle->qids );
  printf(" ,\n    {\n");
  if( tangle->qureg.size > 32 ) {
    printf("<a large quantum state>, really print? (y/N): ");
    /* if( getchar() == 'y' ) */
    /*   quantum_print_qureg( tangle->qureg ); */
  }
  else
    quantum_print_qureg( tangle->qureg );
  printf("}");
}

/***********
 ** QUBIT **
 ***********/
typedef struct qubit {
  tangle_t* tangle;
  qid_t qid;
  pos_t pos;
} qubit_t;

qubit_t _invalid_qubit_ = { NULL, -1, -1 };

// use this function to return a correct qureg position
//  libquantum uses an reverse order (least significant == 0)
int get_target( const qubit_t qubit ) {
  return qubit.tangle->size - qubit.pos - 1;
}

bool invalid( const qubit_t qubit ) {
  return qubit.tangle == NULL;
}
quantum_reg* get_qureg( const qubit_t qubit ) {
  return &qubit.tangle->qureg;
}

cl_mem get_buffer (const qubit_t qubit) {
    return qubit.tangle->qureg_buffer;
}



/**********
 ** QMEM **
 **********/
typedef struct signal_map {
  // two bitfields
  //  entries : if qid has an entry, not needed for correct programs
  //  signals : value of the signal
  unsigned char entries[BITNSLOTS(MAX_QUBITS)];
  unsigned char signals[BITNSLOTS(MAX_QUBITS)];
} signal_map_t;

typedef struct qmem {
  size_t size;
  signal_map_t signal_map;
  tangle_t* tangles[MAX_TANGLES];
} qmem_t;


void print_signal_map( const signal_map_t* restrict signal_map ) {
  printf(" {\n");
  for( int qid=0 ; qid<MAX_QUBITS ; ++qid ) {
    if( BITTEST(signal_map->entries, qid) )
      printf("  %d -> %d,\n", qid,
	     BITTEST(signal_map->signals, qid) ? 1 : 0 );
  }  
  printf(" }\n");
}

bool get_signal( const qid_t qid, 
		 const signal_map_t* restrict signal_map ) {
  if( BITTEST(signal_map->entries,qid) )
    return BITTEST(signal_map->signals, qid);
  else {
    printf( "ERROR: I was asked a signal map entry (qid:%d) that wasn't there,\n\
  check quantum program correctness.\n", qid);
    printf( "   signal map:\n     ");
    print_signal_map( signal_map );
    exit(EXIT_FAILURE);
  }

}
void set_signal( const qid_t qid, 
		 const bool signal, 
		 signal_map_t* restrict signal_map ) {
  if( BITTEST(signal_map->entries, qid) ) {
    printf( "ERROR: I was asked to set an already existing signal,\n\
  check quantum program correctness.\n");
    printf( "   signal map:\n     ");
    print_signal_map( signal_map );
    exit(EXIT_FAILURE);
  }
  BITSET(signal_map->entries, qid);
  if( signal )
    BITSET(signal_map->signals, qid);
}

qubit_t 
find_qubit_in_tangle( const qid_t qid, 
		      const tangle_t* restrict tangle )
{
  assert(tangle);
  qid_list_t* qids = tangle->qids;
  if( tangle->size == 0 ) {
    printf("WARNING: looking for qid in empty tangle, this is not "
	   "supposed to happen (deallocate this tangle)\n");
    return _invalid_qubit_;
  }
  for( int i=0;
       qids;
       ++i, qids=qids->rest ) {
    if( qids->qid == qid ) 
      return (qubit_t){ (tangle_t*)tangle, qid, i };
  }
  return _invalid_qubit_;
}

qubit_t 
find_qubit(const qid_t qid, const qmem_t* restrict qmem) {
  tangle_t* tangle;
  qubit_t qubit;
  for( int i=0, tally=0 ; tally < qmem->size ; ++i ) {
    tangle = qmem->tangles[i];
    if( tangle ) {
      qubit = find_qubit_in_tangle(qid, tangle);
      if( !invalid(qubit) )
	return qubit;      
      ++tally;
    }
  }
  return _invalid_qubit_;
}

qid_list_t* add_qid( const qid_t qid, qid_list_t* restrict qids ) {
  // assuming qid is NOT already in qids
  // ALLOC QUBIT LIST
  qid_list_t* restrict new_qids = (qid_list_t*) malloc(sizeof(qid_list_t));
  new_qids->qid = qid;
  new_qids->rest = qids;
  return new_qids;
}

void append_qids( qid_list_t* new_qids, qid_list_t* target_qids ) {
  assert( new_qids && target_qids );
  while( target_qids->rest ) {
    target_qids = target_qids->rest;
  }
  target_qids->rest = new_qids;  
}

/* void remove_qid( qid_t qid, qid_list_t* restrict qids ) { */
/*   qid_list_t* restrict next; */
/*   if( qids ) { */
/*     next = qids->rest; */
/*     if( next ) { */
/*       if( next->qid == qid ) { */
/* 	qids->rest = next->rest; */
/* 	free( next ); */
/*       } */
/*       else { */
/* 	remove_qid( qid, next ); */
/*       } */
/*     } */
/*   } */
/*   else */
/*     // recursion stops if qid was not found */
/*     printf("Warning: I was asked to remove qid %d from a tangle that did" */
/* 	   "not have it\n", qid); */
/* } */


void print_qmem( const qmem_t* restrict qmem ) {
  assert(qmem);
  printf("qmem has %d tangles:\n  {", (int)qmem->size);
  for( int i=0, tally=0 ; tally < qmem->size ; ++i ) {
    assert(i<MAX_TANGLES);
    if( qmem->tangles[i] ) {
      if( tally>0 )
	printf(",\n   ");
      print_tangle(qmem->tangles[i]);
      ++tally;
    }
  }
  printf("}\n");
  printf("signal map:");
  print_signal_map( &qmem->signal_map );
}

qmem_t* init_qmem() {
  qmem_t* restrict qmem = malloc(sizeof(qmem_t)); //ALLOC qmem

  qmem->size = 0;
  //qmem->tangles = calloc(MAX_TANGLES,sizeof(tangle_t*)); //ALLOC tangles
  //memset(&qmem->tangles, 0, sizeof(tangle_t*) * MAX_TANGLES);
  for( int i=0; i<MAX_TANGLES; ++i )
    qmem->tangles[i] = NULL;
  qmem->signal_map = (signal_map_t){{0},{0}};
  
  // instantiate prototypes (libquantum quregs)
  target_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &ret);
  _proto_diag_qubit_ = quantum_new_qureg(1); 
  _proto_dual_diag_qubit_ = quantum_new_qureg(2);
  general_quantum_CZ(0, 1, &_proto_dual_diag_qubit_);

  quantum_reg test = quantum_new_qureg(3);
  COMPLEX_FLOAT a = {0,0};
  COMPLEX_FLOAT b = {1,0};
  COMPLEX_FLOAT c = {2,0};
  COMPLEX_FLOAT d = {3,0};
  COMPLEX_FLOAT e = {4,0};
  COMPLEX_FLOAT f = {5,0};
  COMPLEX_FLOAT g = {6,0};
  COMPLEX_FLOAT h = {7,0};
  
  test.amplitudes[0] = a;
  test.amplitudes[1] = b;
  test.amplitudes[2] = c;
  test.amplitudes[3] = d;
  test.amplitudes[4] = e;
  test.amplitudes[5] = f;
  test.amplitudes[6] = g;
  test.amplitudes[7] = h;


      
  cl_mem test_buffer = qureg_to_buffer(test); 
  
  parralel_quantum_X(test_buffer, 8, 0);

  quantum_reg output = buffer_to_qureg(test_buffer, 3);

  quantum_print_qureg(output);

  srand(time(0));
  return qmem;

}

void free_qmem(qmem_t* qmem) {


  for( int i=0, tally=0 ; tally < qmem->size ; i++ ) {
    assert(i<MAX_TANGLES);
    if( qmem->tangles[i] ) {
      free_tangle(qmem->tangles[i]);
      qmem->tangles[i] = NULL;
      ++tally;
    }
  }
  //free(qmem->tangles); //FREE tangles
  free(qmem); //FREE qmem
}

tangle_t* get_free_tangle(qmem_t* qmem) {
  tangle_t* restrict new_tangle = init_tangle();
  assert(new_tangle);
  // I loop here because tangles can get de-allocated (NULL-ed)
  for(int i=0; i<MAX_TANGLES; ++i) {
    if( qmem->tangles[i] == NULL ) {
      qmem->tangles[i] = new_tangle;
      return new_tangle;
    }
  }
  printf("Ran out of qmem memory, too many tangles! (> %lu)\n",MAX_TANGLES);
  exit(EXIT_FAILURE);
}

tangle_t*
add_dual_tangle( const qid_t qid1, 
		 const qid_t qid2, 
		 qmem_t* restrict qmem) {
  // allocate new tangle in qmem
  tangle_t*  restrict tangle = get_free_tangle(qmem);

  // init tangle
  tangle->qids = add_qid( qid2, tangle->qids );
  tangle->qids = add_qid( qid1, tangle->qids );
  tangle->size = 2;

  // update qmem info
  qmem->size += 1;

  // init quantum state
  tangle->qureg_buffer = qureg_to_buffer(_proto_dual_diag_qubit_);
  tangle->qureg = quantum_new_qureg(1);
  return tangle;
}

tangle_t* 
add_tangle( const qid_t qid, 
	    qmem_t* restrict qmem ) {
  // allocate new tangle in qmem
  tangle_t*  restrict tangle = get_free_tangle(qmem);
  // init tangle
  tangle->qids = add_qid( qid, tangle->qids );
  tangle->size = 1;
  // update qmem info
  qmem->size += 1;
  // init quantum state
  tangle->qureg_buffer = qureg_to_buffer(_proto_diag_qubit_);
  tangle->qureg = quantum_new_qureg(1);
  return tangle;
}


/* Adds new qubit BEHIND existing state: |q> x |+>
 */
void
add_qubit( const qid_t qid, 
	   tangle_t* restrict tangle) {
  assert(tangle);
  // appends new qid:  qids := [[qids...],qid]
  append_qids( add_qid(qid,NULL), tangle->qids );
  int oldsize = 1 << tangle->size;
  tangle->size += 1;
  // tensor |+> to tangle
  cl_mem _proto_buffer = qureg_to_buffer(_proto_diag_qubit_);
  const cl_mem new_buffer = 
    parralel_kronecker(tangle->qureg_buffer,_proto_buffer, oldsize, 2);
  // out with the old
  ret = clReleaseMemObject(tangle->qureg_buffer);
  // in with the new
  tangle->qureg_buffer = new_buffer;
}

void 
delete_tangle( tangle_t* tangle,
	       qmem_t* restrict qmem ) {
  assert( tangle );
  assert( tangle->qids == NULL );
  qmem->size -= 1;
  // null the tangle entry in qmem
  for(int i=0; i<MAX_TANGLES; ++i) {
    if( qmem->tangles[i] == tangle ) {
      qmem->tangles[i] = NULL;
      free_tangle( tangle );
      return;
    }
  }
  free_tangle( tangle );
  printf("ERROR: I was asked to delete an unknown tangle in qmem\n");
  exit(EXIT_FAILURE);
}

void 
delete_qubit(const qubit_t qubit, 
	     qmem_t* restrict qmem) {
  assert( !invalid(qubit) );
  tangle_t* tangle = qubit.tangle;
  // handle points to where current qid entry is stored
  qid_list_t** handle = &tangle->qids;
  qid_list_t* qids = tangle->qids;
  
  // listref by qubit.pos
  for( int i=0 ; i < qubit.pos ; ++i ) {
    handle = &qids->rest;
    qids = qids->rest;
  }
  assert(qids);
  assert(qids->qid == qubit.qid);

  // pointer plumbing:
  //  bypass current qid entry
  *handle = qids->rest;
  tangle->size -= 1;

  free(qids); // FREE QUBIT LIST element
 // when empty, dealloc tangle
  if( tangle->size == 0 ) {
    tangle->qids = NULL;
    delete_tangle( tangle, qmem );
  }
}

void 
merge_tangles(tangle_t* restrict tangle_1, 
	      tangle_t* restrict tangle_2, 
	      qmem_t* restrict qmem) {
  assert( tangle_1 && tangle_2 );
  int size_1 = pow(2,tangle_1->size);
  int size_2 = pow(2,tangle_2->size);
  tangle_1->size = tangle_1->size + tangle_2->size;
  // append qids of tangle_2 to tangle_1, destructively
  append_qids( tangle_2->qids, tangle_1->qids);
  // tensor both quregs
  const cl_mem new_qureg = 
    parralel_kronecker(tangle_1->qureg_buffer, tangle_2->qureg_buffer, size_1, size_2);
  // out with the old
  ret = clReleaseMemObject(tangle_1->qureg_buffer);
  // in with the new
  tangle_1->qureg_buffer = new_qureg;

  tangle_2->qids = NULL; // avoids the qid_list from being collected
  delete_tangle( tangle_2, qmem ); //free the tangle
}

void ensure_list( sexp_t* exp ) {
  CSTRING* str = NULL;
 
  if( exp->ty != SEXP_LIST ) {
    print_sexp_cstr( &str, exp, STRING_SIZE );
    printf("ERROR: malformed expression, expecting a list expression and\
  got:\n  %s", toCharPtr( str ));
    exit(EXIT_FAILURE);
  }


}

void ensure_value( sexp_t* exp ) {
  CSTRING* str = NULL;

  if ( exp->ty != SEXP_VALUE ) {
    print_sexp_cstr( &str, exp, STRING_SIZE );
    printf("ERROR: malformed expression, expecting a value expression and\
  got:\n  %s", toCharPtr( str ));
    sdestroy( str );
    exit(EXIT_FAILURE);
    sdestroy( str );
  }
}

char get_opname( sexp_t* exp ) {
  return exp->val[0];
}

int get_qid( sexp_t* exp ) {
  return atoi( exp->val );
}

typedef struct angle_constant {
  const char* name;
  double value;
} angle_constant_t;

#define ANGLE_CONSTANT_MAX_CHARS 8
#define ANGLE_CONSTANTS_MAX 32
angle_constant_t _angle_constants_[ANGLE_CONSTANTS_MAX] = {
  {"PI",M_PI},
  {"PI/2",M_PI/2},
  {"PI/4",M_PI/4},
  {"PI/8",M_PI/8},
  {"-PI",-M_PI},
  {"-PI/2",-M_PI/2},
  {"-PI/4",-M_PI/4},
  {"-PI/8",-M_PI/8}
};
int _angle_constants_free_ = 8;

void add_new_constant(const char* name, double value) {
  
  if( _angle_constants_free_ > sizeof(_angle_constants_) ) {
    printf("ERROR: I can only remember %lu angle constants, I was asked"
	   " to add one more\n", sizeof(_angle_constants_));
    exit(EXIT_FAILURE);
  }
  angle_constant_t* restrict entry = 
    &_angle_constants_[_angle_constants_free_++];
  entry->name = name;
  entry->value = value;
}

double lookup_angle_constant(const char* str) {
  if( str )
    for( int i=0; i<_angle_constants_free_; ++i ) {
      if( strcmp(str, _angle_constants_[i].name) == 0 )
	return _angle_constants_[i].value;
    }
  return 0.0;
}


double parse_angle( const sexp_t* exp ) {
  /* syntax:
       <angle>  ::=  (- <angle>) | *angle_constant* | *float*  
  */ 
  //I can be more advanced and add some calc functionality,
  // but I'm not that insane atm. (some lib?)

  if( exp->ty == SEXP_LIST ) {
    // (- <angle>)
    const sexp_t* sign = exp->list;
    if( strcmp(sign->val, "-") == 0 )
      return -parse_angle(sign->next);
    else {
      printf("ERROR: expected (- ...) while parsing angle,"
	     "gotten:%s\n", sign->val);
      exit(EXIT_FAILURE);
    }
  }

  double angle = strtod( exp->val, NULL );
  if( angle == 0.0 && exp->val[0]!='0' ) { // atof failed
    // upper case 'str'
    char str[ANGLE_CONSTANT_MAX_CHARS];
    strcpy( str, exp->val );
    // ensure there is a termination string
    str[ANGLE_CONSTANT_MAX_CHARS-1] = 0;
    for(int i=0; str[i]; ++i) 
      str[i] = toupper(str[i]);

    // maybe it is specified in the environment?
    char* env_result = getenv(str);
    if( env_result ) {
      angle = atof( env_result );
      if( angle != 0.0 )
	return angle;
      else { // perhaps env contains one of our internal constants?
	for(int i=0; env_result[i]; ++i) 
	  env_result[i] = toupper(env_result[i]);
	return lookup_angle_constant(env_result);
      }
    }
    else {
      // maybe it's one of our constants
      angle = lookup_angle_constant(str);
      if( angle != 0.0 ) 
	return angle;
      else {
	// fallthrough: I really don't know what to do with this constant,
	// ask for input to the user
	printf(" angle \"%s\" is not a recognised constant, "
	       "please insert value: \n", str);
	int scanresult = scanf("%lf",&angle);
	if( scanresult ) {
	  printf(" added %s as %lf\n",str,angle);
	  add_new_constant(str, angle); 
        }
        else
          exit(EXIT_FAILURE);
      }
    }
  }
  return angle;
}

/************************
 ** QUANTUM OPERATIONS **
 ************************/
/*
static inline unsigned int
quantum_hash64(MAX_UNSIGNED key, int width)
{
  unsigned int k32;

  k32 = (key & 0xFFFFFFFF) ^ (key >> 32);

  k32 *= 0x9e370001UL;
  k32 = k32 >> (32-width);

  return k32;
}

static inline int
quantum_get_state(MAX_UNSIGNED a, quantum_reg reg)
{
  int i;

  if(!reg.hashw)
    return a;

  i = quantum_hash64(a, reg.hashw);

  while(reg.hash[i])
    {
      if(reg.node[reg.hash[i]-1].state == a)
	return reg.hash[i]-1;
      i++;
      if(i == (1 << reg.hashw))
	i = 0;
    }
  
  return -1;   
}
/* Add an element to the hash table *

static inline void
quantum_add_hash(MAX_UNSIGNED a, int pos, quantum_reg *reg)
{
  int i, mark = 0;

  i = quantum_hash64(a, reg->hashw);

  while(reg->hash[i])
    {
      i++;
      if(i == (1 << reg->hashw))
	{
	  if(!mark)
	    {
	      i = 0;
	      mark = 1;
	    }
	  else
	    quantum_error(QUANTUM_EHASHFULL);
	}
    }

  reg->hash[i] = pos+1;

}

* Reconstruct hash table *

static inline void
quantum_reconstruct_hash(quantum_reg *reg)
{
  int i;

  * Check whether register is sorted *

  if(!reg->hashw)
    return;
  
  for(i=0; i<(1 << reg->hashw); i++)
    reg->hash[i] = 0;
  for(i=0; i<reg->size; i++)
    quantum_add_hash(reg->node[i].state, i, reg);
}

*/

/*
int
quantum_diag_measure(int pos, double angle, quantum_reg* restrict reg)
{


  quantum_reg out = quantum_new_qureg(reg->qubits-1);
  MAX_UNSIGNED pos2 = (MAX_UNSIGNED) 1 << pos;
  double limit = (1.0 / ((MAX_UNSIGNED) 1 << reg->qubits)) / 1000000;
  double prob=0, norm = 0;
  COMPLEX_FLOAT amp = 0;

  typedef unsigned int basis;
  basis upper_mask = ((basis)(-1/pos2))*pos2;
  basis lower_mask = -1 % pos2;
  assert( upper_mask + lower_mask == -1 );
  
  basis lpart,rpart;
  int free = 0;
  //quantum_print_qureg(*reg);
  for(basis state=0; state<(1 << out.qubits); ++state ) {
    lpart = upper_mask & state<<1;
    rpart = lower_mask & state;
    basis k = lpart+rpart;
    int i = k^pos2;
    int k_is_odd = k & pos2;
    if( reg->amplitudes[k] != 0)
      amp += k_is_odd ? -(reg->amplitudes[k] * quantum_cexp(-angle))
	              : reg->amplitudes[k];
    if( reg->amplitudes[i] != 0)
      amp += k_is_odd ? reg->amplitudes[i] 
	              : -(reg->amplitudes[i] * quantum_cexp(-angle));
      prob = quantum_prob_inline( amp );
      if( prob > limit ) {
	    norm += prob;
	    out.amplitudes[state] = amp;
      }
      else out.amplitudes[state] = 0 ;
      amp = 0;
    }
  
  printf("just testing: reg=%u ; out=%u", reg->qubits, out.qubits);
  
  quantum_delete_qureg(reg);
 *reg = out;

 return 1;
 }
*/


void qop_cz( const qubit_t qubit_1, const qubit_t qubit_2 ) {
  const int tar1 = get_target(qubit_1);
  const int tar2 = get_target(qubit_2);
  assert( !(invalid(qubit_1) || invalid(qubit_2)) );
  assert( qubit_1.tangle == qubit_2.tangle );
  int size = pow(2,qubit_1.tangle->size);

  parralel_quantum_CZ(get_buffer(qubit_1), size, tar1, tar2);

  
}

void qop_x( const qubit_t qubit ) {
  assert( !invalid(qubit) );
  size_t size = pow(2,qubit.tangle->size);
  parralel_quantum_X(get_buffer(qubit), size, get_target(qubit));
}

void qop_z( const qubit_t qubit ) {
  assert( !invalid(qubit) );
  int size = pow(2,qubit.tangle->size);
  parralel_quantum_Z(get_buffer(qubit), size, get_target(qubit));
}

/* Apply a phase kick by the angle GAMMA */
/*
void
quantum_inv_phase_kick(int target, double gamma, quantum_reg *reg)
{
  int i;
  COMPLEX_FLOAT z;

  z = quantum_conj(quantum_cexp(gamma));
  double* p = (double*)&z;
  printf("before phase kick (z=%f,%fi):\n",p[0],p[1]);
  //  quantum_print_qureg( *reg );
  
  
  for(i=0; i<reg->size; i++)
    {
      if(i & ((MAX_UNSIGNED) 1 << target))
	reg->amplitudes[i] *= z;
    }

  printf("\nafter phase kick:\n");

  //  quantum_decohere(reg);
}

*/

/***************
 ** EVALUATOR **
 ***************/
void eval_E(sexp_t* exp, qmem_t* qmem) {
  int qid1, qid2;
  qubit_t qubit_1;
  qubit_t qubit_2;

  assert( qmem );

  // move to the first argument
  exp = cdr(exp);
  if( !exp ) {
    printf("Entangle did not have any qubit arguments");
    exit(EXIT_FAILURE);
  }
  qid1 = get_qid( exp );
  
  // move to the second argument
  exp = cdr(exp);
  if( !exp ) {
    printf("Entangle did not have a second argument");
    exit(EXIT_FAILURE);
  }
  qid2 = get_qid( exp );

  // get tangle for qid1
  qubit_1 = find_qubit( qid1, qmem );
  qubit_2 = find_qubit( qid2, qmem );

  if( invalid(qubit_1) )
    if( invalid(qubit_2) ) {
      // if both unknown, create new tangle with two |+> states
      add_dual_tangle(qid1, qid2, qmem);
      return; // already in correct state by construction
    }
    else
      // add qid1 to qid2's tangle
      add_qubit( qid1, qubit_2.tangle );
  else
    if( invalid(qubit_2) )
      // add qid2 to qid1's tangle
      add_qubit( qid2, qubit_1.tangle );
    else
      if( qubit_1.tangle == qubit_2.tangle ) {
	// if not, qubit entries are already valid
	qop_cz( qubit_1, qubit_2 );
	return;
      }
      else
	// both tangles are non-NULL, merge both
	merge_tangles(qubit_1.tangle, qubit_2.tangle, qmem);
  // get valid qubit entries
  qubit_1 = find_qubit( qid1, qmem );
  qubit_2 = find_qubit( qid2, qmem );
  qop_cz( qubit_1, qubit_2 );
}

/* Parses and checks the value of the given signal(s) */
/*   Syntax:  <identifier> | 0 | 1 | (q <qubit>) | (+ {<signal>}+ ) */
bool satisfy_signals( const sexp_t* restrict exp, 
		      const qmem_t* restrict qmem) {
  const sexp_t* args;
  const sexp_t* first_arg;
  bool signal;
  CSTRING* str = NULL;

  if( exp->ty == SEXP_LIST ) {
    args = exp->list;
    if( args->ty == SEXP_VALUE ) {
      if( strcmp(args->val, "q")==0 || 
	  strcmp(args->val, "Q")==0 ||
	  strcmp(args->val, "s")==0 ||
	  strcmp(args->val, "S")==0) {
	first_arg = args->next;
	return get_signal( atoi(first_arg->val), &qmem->signal_map );
      }
      else
	if( strcmp(args->val, "+")==0 ) {
	  first_arg = args->next;
	  signal = satisfy_signals(first_arg, qmem);
	  for(sexp_t* arg=first_arg->next; arg; arg=arg->next) {
	    signal ^= satisfy_signals(arg, qmem);
	  }
	  return signal;
	} 
    }// otherwise, fall through to parse_error
  }
  else
    if( exp->ty == SEXP_VALUE ) {
      if( strcmp(exp->val, "0") == 0 )
	return false;
      if( strcmp(exp->val, "1") == 0 )
	return true;
    } // otherwise, fall through to parse_error

  print_sexp_cstr( &str, exp, STRING_SIZE );
  printf("ERROR: I got confused parsing signal: %s\n", toCharPtr( str ));
  printf("  signal syntax:  <identifier> | 0 | 1 | (q <qubit>) |"
	 " (+ {<signal>}+ )\n");
  sdestroy(str);
  exit(EXIT_FAILURE);
}

void eval_M(sexp_t* exp, qmem_t* qmem) {
  int qid;
  double angle = 0.0;
  tangle_t* tangle;
  int signal;
  assert( qmem );

  // move to the first argument
  exp = cdr(exp);
  if( !exp ) {
    printf("Measurement did not have any target qubit argument\n");
    exit(EXIT_FAILURE);
  }
  qid = get_qid( exp );
  
  // move to the second argument
  exp = cdr(exp);
  if( exp ) { // default is 0
    angle = parse_angle( exp );
    // change angles by s- and t-signals when available
    exp = cdr(exp);
    if( exp ) { //s-signal, flips sign
      if( _verbose_ )
	printf("before angle correction, angle: %f\n", angle);
      if( satisfy_signals(exp, qmem) )
	angle = -angle;
      exp = cdr(exp);
      if( exp )  //t-signal, adds PI to angle
	if( satisfy_signals(exp, qmem) )
	  angle += M_PI;
    }
  }
  
  //  printf("  Measuring qubits %d\n",qid);

  qubit_t qubit = find_qubit( qid, qmem );
  if( invalid(qubit) ) {
    // create new qubit
    tangle = add_tangle( qid, qmem );
    qubit = find_qubit_in_tangle( qid, tangle );
  }
  // libquantum can only measure in ortho basis,
  //  but <+|q = <0|Hq makes it diagonal
  //  and <+_a| = <+|P_-a
  if( _verbose_ )
    printf("  measuring qubit %d on angle %2.4f\n", qid, angle);
  /* printf("   before + correction:\n"); */
  /* quantum_print_qureg( qubit.tangle->qureg ); */
  
  //  quantum_inv_phase_kick( get_target(qubit), angle, get_qureg(qubit) );

  //if( _alt_measure_ )
    cl_mem newbuffer = parralel_diag_measure(get_buffer(qubit), get_target(qubit), angle, qubit.tangle->size);
    signal = 1;
    qubit.tangle->qureg_buffer = newbuffer;

  /*else {
    quantum_phase_kick( get_target(qubit), -angle, get_qureg( qubit ) );
  
    //printf("   after kick: \n");
    //  quantum_print_qureg( qubit.tangle->qureg );

    quantum_hadamard( get_target(qubit), get_qureg( qubit ) );
  
    //printf("   measuring : \n"     );
    // quantum_print_qureg( qubit.tangle->qureg );
    signal = quantum_bmeasure( get_target(qubit), get_qureg( qubit ) );

    //signal = quantum_diag_measure( get_target(qubit), angle, get_qureg(qubit) );
  }
  */
    /* quantum_hadamard( get_target(qubit), get_qureg( qubit ) ); */
  /* quantum_phase_kick( get_target(qubit), angle, get_qureg( qubit ) ); */

  

  /* printf("   result is %d\n",signal); */
  set_signal( qid, signal, &qmem->signal_map );

  // remove measured qubit from memory
  delete_qubit( qubit, qmem );
}


void eval_X(sexp_t* exp, qmem_t* qmem) {
  qid_t qid;
  qubit_t qubit;
  tangle_t* restrict tangle;
  assert( qmem );

  // move to the first argument
  exp = cdr(exp);
  if( !exp ) {
    printf("X-correction did not have any target qubit argument\n");
    exit(EXIT_FAILURE);
  }
  qid = get_qid( exp );

  if( cdr(exp) ) { 
    if( _verbose_ )
      printf(" (signal was: %d)\n", satisfy_signals( cdr(exp), qmem ));
    // there is a signal argument, bail out early if not satisfied
    if( satisfy_signals( cdr(exp), qmem ) == 0)
      return;
  }

  qubit = find_qubit( qid, qmem );
  if( invalid(qubit) ) {
    // create new qubit
    tangle = add_tangle( qid, qmem );
    qubit = find_qubit_in_tangle( qid, tangle );
  }
  qop_x( qubit );
}

void eval_Z(sexp_t* exp, qmem_t* qmem) {
  qid_t qid;
  qubit_t qubit;
  tangle_t* restrict tangle;
  assert( qmem );

  // move to the first argument
  exp = cdr(exp);
  if( !exp ) {
    printf("Z-correction did not have any target qubit argument\n");
    exit(EXIT_FAILURE);
  }
  qid = get_qid( exp );

  if( cdr(exp) ) { 
    // there is a signal argument, bail out early if not satisfied
    if( _verbose_ )
      printf(" (signal was: %d)\n", satisfy_signals( cdr(exp), qmem ));
    if( satisfy_signals( cdr(exp), qmem ) == 0 )
      return;
  }

  qubit = find_qubit( qid, qmem );
  if( invalid(qubit) ) {
    // create new qubit
    tangle = add_tangle( qid, qmem );
    qubit = find_qubit_in_tangle( qid, tangle );
  }
  qop_z( qubit );
}
 
// expects a list, evals the first argument and calls itself tail-recursively
void eval( sexp_t* restrict exp, qmem_t* restrict qmem ) {
  CSTRING* str = snew(0);
  sexp_t* command;
  sexp_t* rest;
  char opname;

  assert( qmem );

  if( exp == NULL )
    return;
  
  //ensure_list( exp );
  if( exp->ty == SEXP_LIST ) {
    command = car(exp);
    rest = cdr(exp);
    if( _verbose_ ) {
      print_sexp_cstr( &str, exp, STRING_SIZE );
      printf("evaluating %s\n", toCharPtr(str));
    }
  }
  else {
    assert( exp->ty == SEXP_VALUE );
    command = exp;
    rest = NULL;
    sexp_t tmp_list = (sexp_t){SEXP_LIST, NULL, 0, 0, exp, NULL, 0,
			       NULL, 0};
    //    print_sexp_cstr( &str, new_sexp_list(exp), STRING_SIZE );
    if( _verbose_ ) {
      print_sexp_cstr( &str, &tmp_list, STRING_SIZE );
      printf("evaluating %s\n", toCharPtr(str));
    }
  }
  
  opname = get_opname( command );

  sdestroy( str );
    
  switch ( opname ) {
  case 'E': 
    eval_E( command, qmem ); 
    if( _verbose_ )
      print_qmem(qmem);
    eval( rest, qmem ); 
    break;
  case 'M': 
    eval_M( command, qmem ); 
    if( _verbose_ )
      print_qmem(qmem);
    eval( rest, qmem ); 
    break;
  case 'X': 
    eval_X( command, qmem ); 
    if( _verbose_ )
      print_qmem(qmem);
    eval( rest, qmem ); 
    break;
  case 'Z': 
    eval_Z( command, qmem );
    if( _verbose_ )
      print_qmem(qmem);
    eval( rest, qmem ); 
    break;
  default: 
    printf("unknown command: %c\n", opname);
  }
}

COMPLEX_FLOAT parse_complex( const char* str ) {
  char* next_str = NULL;
  char* last_str = NULL;
  cl_float real = strtod(str, &next_str);
  cl_float imag = strtod(next_str, &last_str);
  COMPLEX_FLOAT cplx = {real,imag};
  return cplx;
}


void parse_tangle( const sexp_t* exp, qmem_t* restrict qmem ) {
  qubit_t qubit;
  tangle_t* tangle = NULL;
  const sexp_t* qids_exp = exp->list;
  const sexp_t* amps_exp = exp->list->next;

  sexp_to_dotfile(exp,"input.dot");

  assert( qids_exp->ty == SEXP_LIST &&
	  amps_exp->ty == SEXP_LIST );

  const int num_amps = sexp_list_length(amps_exp);

  //at least one qid  
  sexp_t* qids = qids_exp->list;
  assert( qids && qids->val );
  for( sexp_t* qid_exp = qids; qid_exp; qid_exp = qid_exp->next) {
    qubit = find_qubit(get_qid(qid_exp), qmem);
    if( !invalid(qubit) ) {
      fprintf( stderr, 
	       "ERROR: trying to add already existing qubit "
	       "during input file initialization (qid:%d)\n",
	       qubit.qid );
      exit(EXIT_FAILURE);
    }
  }
  
  tangle = get_free_tangle(qmem);
 
  qmem->size += 1;
  tangle->size = sexp_list_length(qids_exp);
  tangle->qureg = quantum_new_qureg(tangle->size);
  tangle->qids = add_qid( get_qid(qids), tangle->qids );
  while(qids->next) {
    qids=qids->next;
    append_qids( add_qid(get_qid(qids), NULL), tangle->qids );
  }

  quantum_reg* reg = &tangle->qureg;
  sexp_t* amp = amps_exp->list;
  for( int i=0; i<num_amps ;  ++i ) {
      if (i == atoi(amp->list->val))
            reg->amplitudes[i] = parse_complex(amp->list->next->val);
    amp=amp->next;
  } 
}

const tangle_t* fetch_first_tangle( const qmem_t* restrict qmem ) {

  for(int i=0; i<MAX_TANGLES; ++i) {
    const tangle_t* tangle = qmem->tangles[i];
    if( tangle )
      return tangle;
  }
  return NULL;
}


/* prints ONLY THE FIRST TANGLE in sexpr form to file, same format as input file, but
   also produces 0's */
void 
produce_output_file( const char* output_file, 
		     const qmem_t* restrict qmem ) {
  CSTRING* out = snew(STRING_SIZE); 
  char str[STRING_SIZE];
  //  int count=0;
  quantum_reg reg;
  const tangle_t* tangle = fetch_first_tangle(qmem);
  assert( tangle );
  assert( output_file );
  saddch(out,'(');
  // print qids
  saddch(out, '(');
  for( const qid_list_t* cons = tangle->qids;
       cons;
       cons=cons->rest ) {
    sprintf(str,"%d", cons->qid);
    sadd(out, str);
    if( cons->rest )
      saddch(out, ' ');
  }
  // end qids
  sadd(out, ")\n ");

  // print (basis amplitude)
  saddch(out, '(');
  reg = tangle->qureg;
  for( int i=0; i<reg.size; ++i ) {
    sprintf(str,"(%lli ", i);
    sadd(out, str);
    sprintf( str, "% .12g%+.12gi)", 
	     real(reg.amplitudes[i]),
	     imag(reg.amplitudes[i]) );
    sadd(out, str);
    if( i+1<reg.size )
      sadd(out, "\n  ");
  }
  // end amplitudes
  saddch(out, ')');
  saddch(out, ')');
  saddch(out, '\n');
  FILE* file = fopen(output_file, "w");
  fputs(toCharPtr(out), file);
  fclose(file);
  sdestroy(out);
}

void initialize_input_state( const char* input_file, qmem_t* qmem ) {
  if( input_file == NULL )
    return;
  int fd = open( input_file, O_RDONLY );
  sexp_iowrap_t* input_port = init_iowrap( fd );
  sexp_t* exp = read_one_sexp(input_port);
  parse_tangle( exp, qmem );
  destroy_sexp( exp );
  destroy_iowrap( input_port );
  close(fd);  
}

//to parralelize, later.

void quantum_normalize( quantum_reg reg ) {
  double limit = 1.0e-8;
  COMPLEX_FLOAT norm={0,0};
  cl_float q_prob;
  for( int i=0; i<reg.size; ++i ) {  
    q_prob = quantum_prob_inline(reg.amplitudes[i]);
    norm = caddfl(norm, q_prob);
  }

  if( abs(1-real(norm)) < limit )
    for( int i=0; i<reg.size; ++i ) {
      reg.amplitudes[i] = cdiv(reg.amplitudes[i], norm);
    }
}

int main(int argc, char* argv[]) {
  sexp_iowrap_t* input_port;
  sexp_t* mc_program;
  start = PAPI_get_real_usec();  
  init_opencl();
  program = build_program("quantum_kernel.cl");
  qmem_t* restrict qmem = init_qmem();
  CSTRING* str = snew( 0 );

  int interactive = 0;
  int silent = 0;
  char* output_file = NULL;
  int program_fd;
  int c;
     
  opterr = 0;
    
  while ((c = getopt (argc, argv, "isvmf:o::")) != -1)
    switch (c)
      {
      case 'i':
	interactive = 1;
	break;
      case 's':
	silent = 1;
	break;
      case 'v':
	_verbose_ = 1;
	break;
      case 'm':
	_alt_measure_ = 1;
	break;
      case 'f':
	initialize_input_state(optarg, qmem);
	break;
      case 'o':
	output_file = optarg;
	break;
      case '?':
	if (optopt == 'f')
	  fprintf (stderr, "Option -%c requires an argument.\n", optopt);
	else if (optopt == 'o') {
	  output_file = "out";
	  break;
	}
	else if (isprint (optopt))
	  fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	else
	  fprintf (stderr, 
		   "Unknown option character `\\x%x'.\n",
		   optopt);
	return 1;
      default:
	abort ();
      }
     
  //  

  if (_verbose_) {
    printf("Initial QMEM:\n ");
    print_qmem( qmem );
  }
  if( interactive ) {
    printf("Starting QVM in interactive mode.\n qvm> ");
    input_port = init_iowrap( 0 );  // we are going to read from stdin
    mc_program = read_one_sexp( input_port );
    while( mc_program ) {
      eval( mc_program->list, qmem );
      print_qmem( qmem );
      printf("\n qvm> ");
      destroy_sexp( mc_program );
      mc_program = read_one_sexp( input_port );
    }
  }
  else {
    // read input program
    program_fd = 
      optind < argc ?                // did the user pass a non-option argument?
      open(argv[optind], O_RDONLY) : // open the file
      0;                             // otherwise, use stdin
    input_port = init_iowrap( program_fd );
    mc_program = read_one_sexp( input_port );
    if( program_fd )
      close( program_fd );
    
    if (!silent) {
      print_sexp_cstr( &str, mc_program, STRING_SIZE );
      printf("I have read: \n%s\n", toCharPtr(str) );
    }
    // emit dot file
    /* sexp_to_dotfile( mc_program->list, "mc_program.dot" ); */
    
    eval( mc_program->list, qmem );
  }

  //get information back from buffers to the quregs
  
  tangle_t* tangle = NULL;
  for(int t=0; t <qmem->size;t++) {
    tangle = qmem->tangles[t];
    if ( tangle ) {
        quantum_reg qureg = buffer_to_qureg(tangle->qureg_buffer, tangle->size);
        tangle->qureg = qureg;
    }
  }
  //normalize at the end, not during measurement

  int tally=0;
  tangle=NULL;
  for( int t=0; tally<qmem->size; ++t ) {
    tangle = qmem->tangles[t];
    if( tangle ) {
      quantum_normalize( tangle->qureg );
      ++tally;
    }
  }

  end = PAPI_get_real_usec();
  
  if (!silent) {
    printf("Resulting quantum memory is:\n");
    print_qmem( qmem );
    printf("Total execution time in ms = %0.3f \n", (end - start) / 1000.0 );

  }

  if( output_file ) {
    produce_output_file(output_file, qmem);
  }
  
  destroy_iowrap( input_port );
  sdestroy( str );
  destroy_sexp( mc_program );
  sexp_cleanup();
  free_qmem( qmem );
  free_opencl();
  return 0;

}

