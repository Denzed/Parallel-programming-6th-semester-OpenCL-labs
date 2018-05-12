__kernel void scan_blelloch(
    __global double *a, 
    __global int n, 
    __global double *r, 
    __local double *b
) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint dp = 1;

    if (gid < n) {
        b[lid] = a[gid];
    }

    for (uint s = block_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) {
            uint i = dp * (2 * lid + 1) - 1;
            uint j = dp * (2 * lid + 2) - 1;
            
            b[j] += b[i];
        }

        dp <<= 1;
    }

    if (lid == 0) {
        b[block_size - 1] = 0;
    }

    for(uint s = 1; s < block_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < s) {
            uint i = dp * (2 * lid + 1) - 1;
            uint j = dp * (2 * lid + 2) - 1;

            int t = b[j];
            b[j] += b[i];
            b[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        r[gid] = b[lid];
    }
}

__kernel void reduce(
    __global int *a, 
    __global int n, 
    __global int *r
) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint dp = 1;

    __local double added_value = 0;
    if (lid == 0 && gid / block_size > 0) {
        added_value = a[gid / block_size - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        r[gid] += added_value;
    }
}