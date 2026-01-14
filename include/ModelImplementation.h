/**
 * ResNet-18 Model Execution
 * Main inference pipeline implementation
 */

#include "Kernel.cuh"
#include "ResNetDev.h"
#include <iostream>

/**
 * @brief Executes ResNet-18 inference pipeline
 * @param model ResNet-18 model with weights loaded on GPU
 * @param image Input image tensor on GPU device memory
 * @return Array of 1000 confidence scores (one per ImageNet class)
 */
float *launchModel(const ResNet18 &model, const float *image)
{
    float *output;
    int outputDim = computeDim(IMAGE_DIM, 2, 3, model.conv1.kernelSize);
    size_t outputSize = outputDim * outputDim * model.conv1.outputSize;

    CHECK_ERROR(cudaMalloc((void **)&output, outputSize * sizeof(float)));
    CHECK_ERROR(cudaMemset(output, 0, outputSize * sizeof(float)));

    // Initial convolution layer
    launchConvKernel((float *)image, output, model.conv1, model.bn1, IMAGE_DIM, 2, 3, true, nullptr);
    cudaDeviceSynchronize();

    // Max pooling: 112x112x64 -> 56x56x64
    float *pooledOutput;
    cudaMalloc(&pooledOutput, 64 * 56 * 56 * sizeof(float));
    launchMaxPoolKernel(output, pooledOutput, 112, 112, 64, 3, 2, 1);
    cudaDeviceSynchronize();

    // Copy pooled output back to main buffer
    cudaMemcpy(output, pooledOutput, 64 * 56 * 56 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(pooledOutput);

    // Layer 1: Two basic blocks (64 channels, 56x56 spatial)
    runBasicBlock(model.layer1[0], output, output, 64, 56, 56, 1);
    cudaDeviceSynchronize();
    runBasicBlock(model.layer1[1], output, output, 64, 56, 56, 1);
    cudaDeviceSynchronize();

    // Layer 2: First block downsamples, second maintains (128 channels, 28x28 spatial)
    runBasicBlock(model.layer2[0], output, output, 64, 56, 56, 2);
    cudaDeviceSynchronize();
    runBasicBlock(model.layer2[1], output, output, 128, 28, 28, 1);
    cudaDeviceSynchronize();

    // Layer 3: First block downsamples, second maintains (256 channels, 14x14 spatial)
    runBasicBlock(model.layer3[0], output, output, 128, 28, 28, 2);
    cudaDeviceSynchronize();
    runBasicBlock(model.layer3[1], output, output, 256, 14, 14, 1);
    cudaDeviceSynchronize();

    // Layer 4: First block downsamples, second maintains (512 channels, 7x7 spatial)
    runBasicBlock(model.layer4[0], output, output, 256, 14, 14, 2);
    cudaDeviceSynchronize();
    runBasicBlock(model.layer4[1], output, output, 512, 7, 7, 1);
    cudaDeviceSynchronize();

    // Adaptive average pooling: 7x7x512 -> 512
    float *globalPooled;
    cudaMalloc(&globalPooled, 512 * sizeof(float));
    launchAdaptiveAvgPoolKernel(output, globalPooled, 7, 7, 512);
    cudaDeviceSynchronize();

    // Fully connected layer: 512 -> 1000 (ImageNet classes)
    float *finalOutput;
    cudaMalloc(&finalOutput, 1000 * sizeof(float));
    launchFCKernel(globalPooled, finalOutput, model.fc, 512, 1000);
    cudaDeviceSynchronize();

    // Copy results back to host
    float *hostResults = new float[1000];
    cudaMemcpy(hostResults, finalOutput, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup GPU memory
    cudaFree(globalPooled);
    cudaFree(finalOutput);
    cudaFree(output);
    return hostResults;
};