// Buffer-based implementation
__kernel void fanbeam_reflection_buffer(__global const float* sinogram,
                                       __global float* output,
                                       int height, int halfwidth,
                                       float SDD, float pixelSize, float axis_pixelposition) {
    int u_right = get_global_id(0);
    int i = get_global_id(1);

    // Prevent out-of-bounds access
    if (u_right >= halfwidth || i >= height) return;

    int u_left = halfwidth - u_right - 1;
    float angle = i * 2 * M_PI / height;
    float b_inright = angle - atan2((u_right - axis_pixelposition) * pixelSize, SDD);

    // Wrap angle to [0, 2π]
    if (b_inright < 0) b_inright += 2 * M_PI;

    // Nearest neighbor interpolation
    int idx = (int)(round(b_inright * height / (2 * M_PI))) % height;

    // Additional boundary check
    if (idx >= 0 && idx < height) {
        output[i * halfwidth + u_right] = sinogram[idx * halfwidth + u_left];
    }
}

// Image-based implementation
__kernel void fanbeam_reflection_image(read_only image2d_t sinogram,
                                       write_only image2d_t output,
                                       int height, int halfwidth,
                                       float SDD, float pixelSize, float axis_pixelposition) {
    int u_right = get_global_id(0);
    int i = get_global_id(1);

    // Prevent out-of-bounds memory access
    if (u_right >= halfwidth || i >= height) return;

    int u_left = halfwidth - u_right - 1;
    float angle = i * 2 * M_PI / height;
    float b_inright = angle - atan2((u_right - axis_pixelposition) * pixelSize, SDD);

    // Wrap angle to [0, 2π]
    if (b_inright < 0) b_inright += 2 * M_PI;

    // Nearest neighbor interpolation
    int idx = (int)(round(b_inright * height / (2 * M_PI))) % height;

    // Additional boundary check
    if (idx >= 0 && idx < height) {
        float2 coord = (float2)((float)u_left / (float)halfwidth, (float)idx / (float)height);
        float value = read_imagef(sinogram, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP, coord).x;
        write_imagef(output, (int2)(u_right, i), value);
    }
}
