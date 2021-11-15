#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include <omp.h>

using namespace std;

template <class T>
T* axpy_opencl(int length, T _a, T* data, int _incx, T* result, int _incy, cl_device_id device, size_t block_size) {
  cl_context context;
  cl_command_queue command_queue;
  cl_int ret;
  cl_int err_code;
  cl_mem input, output;

  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
  command_queue = clCreateCommandQueue(context, device, 0, &ret);

  std::ifstream f("axpy.cl");
  std::stringstream ss;
  ss << f.rdbuf();
  std::string str = ss.str();
  const char* source = str.c_str();
  size_t source_length = str.length();

  input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * length, nullptr, &ret);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * length, nullptr, &ret);
  err_code = clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(T) * length, data, 0, nullptr, nullptr);
  err_code = clEnqueueWriteBuffer(command_queue, output, CL_TRUE, 0, sizeof(T) * length, result, 0, nullptr, nullptr);
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
  err_code = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

  cl_kernel kernel;
  string type(typeid(T).name());
  if (type == "float")
    kernel = clCreateKernel(program, "saxpy", &ret);
  else
    kernel = clCreateKernel(program, "daxpy", &ret);

  int n = length - 1;
  T a = _a;
  int incx = _incx;
  int incy = _incy;

  err_code = clSetKernelArg(kernel, 0, sizeof(int), &n);
  err_code = clSetKernelArg(kernel, 1, sizeof(T), &a);
  err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input);
  err_code = clSetKernelArg(kernel, 3, sizeof(int), &incx);
  err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);
  err_code = clSetKernelArg(kernel, 5, sizeof(int), &incy);

  size_t size = length;
  size_t group_size = block_size;
  T* res = new T[length];

  double start = omp_get_wtime();
  err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &size, &group_size, 0, nullptr, nullptr);
  clFinish(command_queue);
  double end = omp_get_wtime();
  if (type == "float")
    cout << "saxpy result: " << end - start << " \n";
  else 
    cout << "daxpy result: " << end - start << " \n";
  err_code = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(T) * length, res, 0, nullptr, nullptr);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return res;
}

template <class T>
T* axpy(int length, T _a, T* data, int _incx, T* result, int _incy) {
  T* res = new T[length];
  for (int i = 0; i < length; i++)
    res[i * _incx] = result[i * _incx] + _a * data[i * _incy];

  return res;
}

template <class T>
T* axpy_parallel(int length, T _a, T* data, int _incx, T* result, int _incy) {
  T* res = new T[length];
#pragma omp parallel for num_threads(8)
  for (int i = 0; i < length; i++)
    res[i * _incx] = result[i * _incx] + _a * data[i * _incy];

  return res;
}

template <class T>
void print_arr(int length, T* data) {
  for (int i = 0; i < length; i++)
    cout << data[i] << " ";
  cout << endl;
}

bool array_equality(int length, float* data1, float* data2) {
  for (int i = 0; i < length; i++) {
    if (data1[i] != data2[i])
      return false;
  }

  return true;
}

int main()
{
  double start, end;
  const cl_int length = 67108864;
  size_t block_size = 16;
  float* data1 = new float[length];
  float* result1 = new float[length];
  double* data2 = new double[length];
  double* result2 = new double[length];

  for (int i = 0; i < length; i++) {
    data1[i] = i;
    result1[i] = 1;
    data2[i] = i;
    result2[i] = 1;
  }

  cl_uint platform_count = 0;
  cl_device_id device;
  // GET PLATFORMS:
  clGetPlatformIDs(0, nullptr, &platform_count);
  cl_platform_id* platform = new cl_platform_id[platform_count];
  clGetPlatformIDs(platform_count, platform, nullptr);

  // OUTPUT FOR SYNC
  cout << "SYNC VERSION:\n";
  start = omp_get_wtime();
  float* r1 = axpy<float>(length, 2, data1, 1, result1, 1);
  end = omp_get_wtime();
  float* check_r1 = r1;
  cout << "saxpy result: " << end - start << " \n";

  start = omp_get_wtime();
  double* r2 = axpy<double>(length, 2, data2, 1, result2, 1);
  end = omp_get_wtime();
  cout << "daxpy result: " << end - start << " \n\n";

  // OUTPUT FOR OPENCL GPU
  clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  char platformName[128];
  clGetPlatformInfo(platform[0], CL_PLATFORM_NAME, 128, platformName, nullptr);
  cout << platformName << ":\n";

  //start = omp_get_wtime();
  r1 = axpy_opencl<float>(length, 2, data1, 1, result1, 1, device, block_size);
  //end = omp_get_wtime();
  //cout << "saxpy result: " << end - start << " \n";

  //start = omp_get_wtime();
  r2 = axpy_opencl<double>(length, 2, data2, 1, result2, 1, device, block_size);
  //end = omp_get_wtime();
  //cout << "daxpy result: " << end - start << " \n";
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  //OUTPUT FOR OPENCL CPU
  clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  platformName[128];
  clGetPlatformInfo(platform[2], CL_PLATFORM_NAME, 128, platformName, nullptr);
  cout << platformName << ":\n";

  //start = omp_get_wtime();
  r1 = axpy_opencl<float>(length, 2, data1, 1, result1, 1, device, block_size);
  //end = omp_get_wtime();
  //cout << "saxpy result: " << end - start << " \n";

  //start = omp_get_wtime();
  r2 = axpy_opencl<double>(length, 2, data2, 1, result2, 1, device, block_size);
  //end = omp_get_wtime();
  //cout << "daxpy result: " << end - start << " \n";
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;


  // OUTPUT FOR OPENMP
  cout << "OPENMP VERSION:\n";
  start = omp_get_wtime();
  r1 = axpy_parallel<float>(length, 2, data1, 1, result1, 1);
  end = omp_get_wtime();
  cout << "saxpy result: " << end - start << " \n";

  start = omp_get_wtime();
  r2 = axpy_parallel<double>(length, 2, data2, 1, result2, 1);
  end = omp_get_wtime();
  cout << "daxpy result: " << end - start << " \n";

  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;


  system("pause");

  return 0;
}
