#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include <omp.h>

using namespace std;

_cl_image_desc create_image_desc(int width, int height, int size) {
  _cl_image_desc img_desc;
  memset(&img_desc, '\0', sizeof(cl_image_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  img_desc.image_width = width;
  img_desc.image_height = height;

  return img_desc;
}

cl_image_format create_image_format() {
  cl_image_format img_format;
  memset(&img_format, '\0', sizeof(cl_image_format));
  img_format.image_channel_order = CL_R;
  img_format.image_channel_data_type = CL_FLOAT;

  return img_format;
}

template <class T>
T* mult_opencl(size_t sizeX, size_t sizeY, T* _a, T* _b, T* result, cl_device_id device, size_t block_size) {
  cl_context context;
  cl_command_queue command_queue;
  cl_int ret, err_code;
  cl_mem A, B, output;

  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
  command_queue = clCreateCommandQueue(context, device, 0, &ret);

  std::ifstream f("compute.cl");
  std::stringstream ss;
  ss << f.rdbuf();
  std::string str = ss.str();
  const char* source = str.c_str();
  size_t source_length = str.length();

  A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * sizeX * sizeY, nullptr, &ret);
  B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * sizeX * sizeY, nullptr, &ret);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * sizeX * sizeY, nullptr, &ret);
  err_code = clEnqueueWriteBuffer(command_queue, A, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, _a, 0, nullptr, nullptr);
  err_code = clEnqueueWriteBuffer(command_queue, B, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, _b, 0, nullptr, nullptr);
  err_code = clEnqueueWriteBuffer(command_queue, output, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, result, 0, nullptr, nullptr);
  
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
  string buildOp = "-D BS=" + to_string(block_size);
  err_code = clBuildProgram(program, 1, &device, buildOp.c_str(), nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "computeBuf", &ret);
  string type(typeid(T).name());
  //if (type == "float")
    //kernel = 

  int n = sizeX;
  int m = sizeY;
  err_code = clSetKernelArg(kernel, 0, sizeof(int), &m);
  err_code = clSetKernelArg(kernel, 1, sizeof(int), &n);
  err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &A);
  err_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &B);
  err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);

  size_t global_work_size[] = { n, n };
  size_t local_work_size[] = { block_size, block_size };
  T* res = new T[sizeX * sizeY];

  double start = omp_get_wtime();
  err_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
  clFinish(command_queue);
  double end = omp_get_wtime();
  cout << "result: " << end - start << " \n";
  err_code = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, res, 0, nullptr, nullptr);

  clReleaseMemObject(A);
  clReleaseMemObject(B);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return res;
}

template <class T>
T* mult_opencl_optimized(size_t sizeX, size_t sizeY, T* _a, T* _b, T* result, cl_device_id device, size_t block_size) {
  cl_context context;
  cl_command_queue command_queue;
  cl_int ret, err_code;
  cl_mem A, B, output;

  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
  command_queue = clCreateCommandQueue(context, device, 0, &ret);

  std::ifstream f("compute.cl");
  std::stringstream ss;
  ss << f.rdbuf();
  std::string str = ss.str();
  const char* source = str.c_str();
  size_t source_length = str.length();

  A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * sizeX * sizeY, nullptr, &ret);
  B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * sizeX * sizeY, nullptr, &ret);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * sizeX * sizeY, nullptr, &ret);
  err_code = clEnqueueWriteBuffer(command_queue, A, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, _a, 0, nullptr, nullptr);
  err_code = clEnqueueWriteBuffer(command_queue, B, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, _b, 0, nullptr, nullptr);
  err_code = clEnqueueWriteBuffer(command_queue, output, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, result, 0, nullptr, nullptr);

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
  string buildOp = "-D BS=" + to_string(block_size);
  err_code = clBuildProgram(program, 1, &device, buildOp.c_str(), nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "computeOptimized", &ret);
  string type(typeid(T).name());
  //if (type == "float")
    //kernel = 

  int n = sizeX;
  int m = sizeY;
  err_code = clSetKernelArg(kernel, 0, sizeof(int), &m);
  err_code = clSetKernelArg(kernel, 1, sizeof(int), &n);
  err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &A);
  err_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &B);
  err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);

  size_t global_work_size[] = { n, n };
  size_t local_work_size[] = { block_size, block_size };
  T* res = new T[sizeX * sizeY];

  double start = omp_get_wtime();
  err_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
  clFinish(command_queue);
  double end = omp_get_wtime();
  cout << "result: " << end - start << " \n";
  err_code = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(T) * sizeX * sizeY, res, 0, nullptr, nullptr);

  clReleaseMemObject(A);
  clReleaseMemObject(B);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return res;
}

template <class T>
T* mult_opencl_image(size_t sizeX, size_t sizeY, T* _a, T* _b, T* result, cl_device_id device, size_t block_size) {
  cl_context context;
  cl_command_queue command_queue;
  cl_int ret, err_code;
  cl_mem A, B, output;

  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
  command_queue = clCreateCommandQueue(context, device, 0, &ret);

  std::ifstream f("compute.cl");
  std::stringstream ss;
  ss << f.rdbuf();
  std::string str = ss.str();
  const char* source = str.c_str();
  size_t source_length = str.length();

  const cl_image_format format = create_image_format();
  const _cl_image_desc desc = create_image_desc(sizeX, sizeY, sizeof(T));
  A = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &ret);
  B = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &ret);
  output = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, nullptr, &ret);
  err_code = clEnqueueWriteImage(command_queue, A, CL_TRUE, new size_t[]{ 0, 0, 0 }, new size_t[]{ sizeX, sizeY, 1 },
    0, 0, _a, 0, nullptr, nullptr);
  err_code = clEnqueueWriteImage(command_queue, B, CL_TRUE, new size_t[]{ 0, 0, 0 }, new size_t[]{ sizeX, sizeY, 1 },
    0, 0, _b, 0, nullptr, nullptr);
  err_code = clEnqueueWriteImage(command_queue, output, CL_TRUE, new size_t[]{ 0, 0, 0 }, new size_t[]{ sizeX, sizeY, 1 },
    0, 0, result, 0, nullptr, nullptr);

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&source_length, &ret);
  string buildOp = "-D BS=" + to_string(block_size);
  err_code = clBuildProgram(program, 1, &device, buildOp.c_str(), nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "computeImage", &ret);
  string type(typeid(T).name());
  //if (type == "float")
    //kernel = 

  int n = sizeX;
  int m = sizeY;
  err_code = clSetKernelArg(kernel, 0, sizeof(int), &m);
  err_code = clSetKernelArg(kernel, 1, sizeof(int), &n);
  err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &A);
  err_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &B);
  err_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);

  size_t global_work_size[] = { n, n };
  size_t local_work_size[] = { block_size, block_size };
  T* res = new T[sizeX * sizeY];

  double start = omp_get_wtime();
  err_code = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
  clFinish(command_queue);
  double end = omp_get_wtime();
  cout << "result: " << end - start << " \n";
  err_code = clEnqueueReadImage(command_queue, output, CL_TRUE, new size_t[]{ 0, 0, 0 }, new size_t[]{ sizeX, sizeY, 1 },
    0, 0, res, 0, nullptr, nullptr);

  clReleaseMemObject(A);
  clReleaseMemObject(B);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return res;
}

template <class T>
T* mult(int sizeX, int sizeY, T* A, T* B, T* result) {
  T* res = new T[sizeX * sizeY];

  /*for (int i = 0; i < sizeX; ++i)
  {
    T* c = res + i * sizeX;
    for (int j = 0; j < sizeX; ++j)
      c[j] = 0;
    for (int k = 0; k < sizeX; ++k)
    {
      const T* b = B + k * sizeX;
      T a = A[i * sizeX + k];
      for (int j = 0; j < sizeX; ++j)
        c[j] += a * b[j];
    }
  }*/
  for (int i = 0; i < sizeX; i++) {
    for (int j = 0; j < sizeX; j++) {
      res[i * sizeX + j] = 0;
      for (int k = 0; k < sizeY; k++) {
        res[i * sizeX + j] += A[i * sizeX + k] * B[j + sizeX * k];
      }
    }
  }

  return res;
}

template <class T>
T* mult_parallel(int sizeX, int sizeY, T* A, T* B, T* result) {
  T* res = new T[sizeX * sizeY];

#pragma omp parallel for num_threads(8)
  for (int i = 0; i < sizeX; i++) {
    for (int j = 0; j < sizeX; j++) {
      res[i * sizeX + j] = 0;
      for (int k = 0; k < sizeY; k++) {
        res[i * sizeX + j] += A[i * sizeX + k] * B[j + sizeX * k];
      }
    }
  }

  return res;
}

template <class T>
T* mult_parallel_optimized(int sizeX, int sizeY, int block_size, T* A, T* B, T* result) {
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < sizeX / block_size; i++)
    for (int j = 0; j < sizeY / block_size; j++)
      for (int n = i * block_size; n < i * block_size + block_size; n++)
        for (int m = j * block_size; m < j * block_size + block_size; m++) {
          for (int k = 0; k < sizeY; k++)
            result[n * sizeY + k] += A[n * sizeY + m] * B[m * sizeY + k];
        }

  return result;
}

template <class T>
void print_arr(int length, T* data) {
  for (int i = 0; i < length; i++)
    cout << data[i] << " ";
  cout << endl;
}

template <class T>
T* clear_array(int length, T* data) {
  for (int i = 0; i < length; i++)
    data[i] = 0;

  return data;
}

template <class T>
bool array_equality(int length, T* data1, T* data2) {
  for (int i = 0; i < length; i++) {
    if (data1[i] != data2[i])
      return false;
  }

  return true;
}

int main()
{
  double start, end;
  const size_t length = 640000;
  size_t size = sqrt(length);
  size_t block_size = 16;
  float* data1 = new float[length];
  float* result = new float[length];
  float* data2 = new float[length];

  for (int i = 0; i < length; i++) {
    data1[i] = rand() % 50;
    result[i] = 0;
    data2[i] = rand() % 50;
  }

  cl_uint platform_count = 0;
  cl_device_id device;
  // GET PLATFORMS:
  clGetPlatformIDs(0, nullptr, &platform_count);
  cl_platform_id* platform = new cl_platform_id[platform_count];
  clGetPlatformIDs(platform_count, platform, nullptr);
  
  char cudaGPU[128], intelGPU[128], intelCPU[128];
  clGetPlatformInfo(platform[0], CL_PLATFORM_NAME, 128, cudaGPU, nullptr);
  clGetPlatformInfo(platform[1], CL_PLATFORM_NAME, 128, intelGPU, nullptr);
  clGetPlatformInfo(platform[2], CL_PLATFORM_NAME, 128, intelCPU, nullptr);


  cout << "---Task 1:\n";


  // OUTPUT FOR SEQ
  cout << "SEQ VERSION:\n";
  start = omp_get_wtime();
  float* r1 = mult<float>(size, size, data1, data2, result);
  end = omp_get_wtime();
  float* check_r1 = r1;
  cout << "result: " << end - start << " \n\n";
  
  // OUTPUT FOR OPENCL GPU
  clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << cudaGPU << ":\n";
  r1 = mult_opencl<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << intelGPU << ":\n";
  r1 = mult_opencl<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;
  
  // OUTPUT FOR OPENCL CPU
  clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << intelCPU << ":\n";
  r1 = mult_opencl<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  // OUTPUT FOR OPENMP
  cout << "OPENMP VERSION:\n";
  start = omp_get_wtime();
  r1 = mult_parallel<float>(size, size, data1, data2, result);
  end = omp_get_wtime();
  cout << "result: " << end - start << " \n";
  if (array_equality(length, check_r1, r1))
    cout << "OK \n" << endl;


  cout << "---Task 2:\n";


  // OUTPUT FOR OPENCL GPU
  clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << cudaGPU << ":\n";
  r1 = mult_opencl_optimized<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << intelGPU << ":\n";
  r1 = mult_opencl_optimized<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  // OUTPUT FOR OPENCL CPU
  clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << intelCPU << ":\n";
  r1 = mult_opencl_optimized<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  // OUTPUT FOR OPENMP
  cout << "OPENMP VERSION:\n";
  start = omp_get_wtime();
  r1 = mult_parallel_optimized<float>(size, size, block_size, data1, data2, result);
  end = omp_get_wtime();
  cout << "result: " << end - start << " \n";
  if (array_equality(length, check_r1, r1))
    cout << "OK \n" << endl;


  cout << "---Task 3:\n";


  // OUTPUT FOR OPENCL GPU
  clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << cudaGPU << ":\n";
  r1 = mult_opencl_image<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << intelGPU << ":\n";
  r1 = mult_opencl_image<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  // OUTPUT FOR OPENCL CPU
  clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  cout << intelCPU << ":\n";
  r1 = mult_opencl_image<float>(size, size, data1, data2, result, device, block_size);
  if (array_equality(length, check_r1, r1))
    cout << "OK" << endl << endl;

  system("pause");

  return 0;
}
