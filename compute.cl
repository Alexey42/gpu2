__kernel void computeBuf(const int m, const int n, __global float *A, __global float *B,
	__global float *result) {
  int x = get_global_id(1);
  int y = get_global_id(0);

  for (int i = 0; i < n; i++) {
	  result[x * n + y] += A[x * n + i] * B[y + n * i];
  }
}

__kernel void computeOptimized(const int m, const int n, __global float *A,
	__global float *B, __global float *result) {
	int xG = get_global_id(1); 
	int yG = get_global_id(0); 
	int xL = get_local_id(1); 
	int yL = get_local_id(0); 
	__local float Al[BS][BS]; 
	__local float Bl[BS][BS]; 
	
	int part = m / BS; 
	result[xG * n + yG] = 0; 
	for (int p = 0; p < part; p++) { 
	  int xp = p * BS + xL; 
	  int yp = p * BS + yL; 
	  Al[xL][yL] = A[xG * m + yp]; 
	  Bl[xL][yL] = B[xp * n + yG]; 
	  barrier(CLK_LOCAL_MEM_FENCE); 
	  for (int t = 0; t < BS; t++) { 
		result[xG * n + yG] += Al[xL][t] * Bl[t][yL]; 
	  } 
	  barrier(CLK_LOCAL_MEM_FENCE); 
	}
}

__kernel void computeImage(const int m, const int n, __read_only image2d_t A, 
	__read_only image2d_t B, __write_only image2d_t result) {
  int xG = get_global_id(1);
  int yG = get_global_id(0);
  int xL = get_local_id(1);
  int yL = get_local_id(0);
  __local float Al[BS][BS];
  __local float Bl[BS][BS];

  float sum = 0;
  int part = n / BS;
  for (int p = 0; p < part; p++) {
	  int xp = p * BS + xL; 
	  int yp = p * BS + yL; 
	  int2 posA = { yp, xG };
	  int2 posB = { yG, xp };
	  Al[xL][yL] = read_imagef(A, posA).x;
	  Bl[xL][yL] = read_imagef(B, posB).x;
	  barrier(CLK_LOCAL_MEM_FENCE);
	  for (int t = 0; t < BS; t++) {
		  sum += Al[xL][t] * Bl[t][yL];
	  }
	  barrier(CLK_LOCAL_MEM_FENCE);
  }
  int2 posC = {yG, xG};
  write_imagef(result, posC, sum);
}