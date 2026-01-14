/**
 * Image Preprocessing Implementation
 * Resize, crop, normalize images for ResNet-18 input format
 */

#include "ImageParse.h"

ImageParse::ImageParse(std::string path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

    // Validate image loading
    if (image.empty())
    {
        std::cerr << "Error: Failed to load image from path: " << path << std::endl;
        return;
    }

    // Convert BGR to RGB
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    // Resize to 256x256 (standard ImageNet preprocessing)
    cv::Mat resizedImage;
    cv::resize(rgbImage, resizedImage, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

    // Center crop to 224x224
    int cropOffsetX = (256 - 224) / 2;
    int cropOffsetY = (256 - 224) / 2;
    cv::Rect cropRegion(cropOffsetX, cropOffsetY, 224, 224);
    cv::Mat croppedImage = resizedImage(cropRegion);

    // Normalize pixel values to [0, 1]
    croppedImage.convertTo(croppedImage, CV_32FC3, 1.0 / 255.0);

    // ImageNet normalization (mean and std)
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdDev[3] = {0.229f, 0.224f, 0.225f};

    // Rearrange to CHW format (channels first) and apply normalization
    for (int h = 0; h < 224; ++h)
    {
        auto row = croppedImage.ptr<cv::Vec3f>(h);
        for (int w = 0; w < 224; ++w)
        {
            size_t spatialIdx = h * 224 + w;
            host[0 * 224 * 224 + spatialIdx] = (row[w][0] - mean[0]) / stdDev[0]; // Red channel
            host[1 * 224 * 224 + spatialIdx] = (row[w][1] - mean[1]) / stdDev[1]; // Green channel
            host[2 * 224 * 224 + spatialIdx] = (row[w][2] - mean[2]) / stdDev[2]; // Blue channel
        }
    }
}