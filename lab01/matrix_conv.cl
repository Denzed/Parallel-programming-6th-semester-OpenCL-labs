__kernel void matrix_conv(
    __global float *a, 
    int n,
    __global float *b, 
    int m,
    __global float *c
) {
    // get coordinates
    int i    = get_global_id(0);
    int j    = get_global_id(1);
    int i_wg = get_local_id(0);
    int j_wg = get_local_id(1);

    __local float b_local[M][M];
    // load b to local memory for faster access inside work group
    b_local[i_wg][j_wg] = b[i_wg * m + j_wg];

    // wait for b to load
    barrier(CLK_LOCAL_MEM_FENCE);

    // check that we are computing value inside C
    if (i < n && j < n) {
        const int HM = (m - 1) >> 1;
    
        // calculate result
        float result = 0;

        for (int k = -HM; k <= HM; ++k) {
            for (int l = -HM; l <= HM; ++l) {   
                // calculate coordinates of A cell
                int a_cell_i = i + k;
                int a_cell_j = j + l;
                // get A cell 
                float a_cell = 0;
                if (0 <= a_cell_i && a_cell_i < n && 
                        0 <= a_cell_j && a_cell_j < n) {
                    a_cell = a[a_cell_i * n + a_cell_j];
                }
                result += a_cell * b_local[k + HM][l + HM];
            }
        }
    
        c[i * n + j] = result;
    }
}