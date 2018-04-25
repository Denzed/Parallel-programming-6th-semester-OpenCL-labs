__kernel void matrix_conv(
   __global float *a, 
   int n,
   __global float *b, 
   int m,
   __global float *c
) {
   int i = get_global_id(0);
   int j = get_global_id(1);
   int k = get_local_id(0);
   int l = get_local_id(1);

   __local float a_local[2 * HM + 1][2 * HM + 1];
   __local float b_local[m][m];

   int a_cell_i = i + k - HM;
   int a_cell_j = j + l - HM;
   
   a_local[k][l] = (
         0 <= a_cell_i && a_cell_i < n && 
         0 <= a_cell_j && a_cell_j < n 
      ? 
         a[a_cell_i * n + a_cell_j] 
      : 
         0
   );

   b_local[k][l] = b[k * m + l];

   barrier(CLK_LOCAL_MEM_FENCE);

   c[i * n + j] += a_local[k][l] * b_local[k][l];
}