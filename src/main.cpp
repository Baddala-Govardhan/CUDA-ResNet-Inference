/**
 * CUDA ResNet-18 Inference Engine
 * Benchmarking and performance evaluation for image classification
 */

#include "ImageParse.h"
#include "Kernel.cuh"
#include "ModelImplementation.h"
#include "ModelParse.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <fmt/format.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int sampleSize = 1000;
constexpr int imgSize = 224 * 224 * 3;
constexpr size_t byteSize = imgSize * sizeof(float);

int main()
{
    // Load and initialize ResNet-18 model
    ModelParse modelParser("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
    ResNet18 model = modelParser.generateModel();

    // Load images onto CPU memory
    std::vector<float> hostImages(imgSize * sampleSize);
    for (int i = 0; i < sampleSize; i++)
    {
        std::string imagePath = fmt::format("assets/cifar10/images/image_{:04d}.png", i);
        ImageParse imageParser(imagePath);

        float *imageData = imageParser.getHostImage();
        std::memcpy(&hostImages[i * imgSize], imageData, byteSize);
    }
    std::cout << "Loaded " << sampleSize << " images\n";

    // Load ImageNet class labels
    std::unordered_map<int, std::string> classLabels;
    std::ifstream classFile("assets/imagenet_classes.txt");
    std::string line;
    int classIdx = 0;

    while (std::getline(classFile, line))
    {
        classLabels[classIdx] = line;
        classIdx++;
    }
    classFile.close();

    // Transfer images to GPU memory
    float *deviceImages;
    CHECK_ERROR(cudaMalloc((void **)&deviceImages, byteSize * sampleSize));
    cudaMemcpy(deviceImages, hostImages.data(), byteSize * sampleSize, cudaMemcpyHostToDevice);

    // GPU warmup iterations
    for (int i = 0; i < 50; i++)
    {
        float *confidenceScores = launchModel(model, deviceImages + (i * imgSize));
        delete[] confidenceScores;
    }

    // Run performance benchmarking
    double totalTime = 0.0;
    for (int i = 0; i < sampleSize; i++)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        float *confidenceScores = launchModel(model, deviceImages + (i * imgSize));
        
        // Find predicted class with highest confidence
        int predictedClass = 0;
        float maxConfidence = confidenceScores[0];
        for (int j = 1; j < 1000; j++)
        {
            if (confidenceScores[j] > maxConfidence)
            {
                maxConfidence = confidenceScores[j];
                predictedClass = j;
            }
        }
        
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime += elapsed.count();

        // Display inference results
        std::string imageName = fmt::format("image_{:04d}", i);
        std::cout << fmt::format("{:<12} Class: {:<30} Confidence: {:>8.4f}  Time: {:>7.2f} ms\n", 
                                  imageName, classLabels[predictedClass], maxConfidence, elapsed.count());
        
        delete[] confidenceScores;
    }

    // Print performance summary
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "Total Time: " << totalTime << " ms\n"
              << "Average Time: " << totalTime / sampleSize << " ms/img\n"
              << "Throughput: " << 1000 * sampleSize / totalTime << " images/sec\n";

    // Cleanup GPU memory
    modelParser.freeModel(model);
    cudaFree(deviceImages);
    return 0;
}