// Lab1.cpp : Defines the entry point for the application.
//

#include "Lab1.h"


int main()
{
	ad::ImgPro imgPro;
	std::string inputFileName = "E:/HW/ImageProcessing/Lab1/Input/sar_1_gray.jpg";
	cv::Mat inputImage = cv::imread(inputFileName, cv::IMREAD_GRAYSCALE);

	std::filesystem::create_directory("Results");

	cv::Mat histImage = imgPro.imageHistogramGray(inputImage);
	cv::imwrite("Results/histogram.jpg", histImage);

	cv::Mat gammaCorImage = imgPro.gammaCorrection(inputImage, 1.1);
	cv::imwrite("Results/gamma.jpg", gammaCorImage);

	std::cout << "Gamma correction MSE = " << imgPro.MSE(inputImage, gammaCorImage) << std::endl;
	std::cout << "Gamma correction SSIM = " << imgPro.MS_SSIM(inputImage, gammaCorImage) << std::endl;

	cv::Mat equalizedImage;
	cv::equalizeHist(inputImage, equalizedImage);
	cv::Mat statisticallyCorrectedImage = imgPro.statisticalColorCorrection(equalizedImage, inputImage);
	cv::imwrite("Results/statisticallyCorrectedImage.jpg", statisticallyCorrectedImage);

	std::cout << "Statistically corrected image MSE = " << imgPro.MSE(inputImage, statisticallyCorrectedImage) << std::endl;
	std::cout << "Statistically corrected image SSIM = " << imgPro.MS_SSIM(inputImage, statisticallyCorrectedImage) << std::endl;

	unsigned thresholdValue = 128;
	unsigned maxValue = 255;
	cv::Mat binarizedImage = inputImage.clone();
	cv::threshold(inputImage, binarizedImage, thresholdValue, maxValue, cv::THRESH_BINARY);

	cv::imwrite("Results/binarizedImage_simple_0.jpg", binarizedImage);

	thresholdValue = 32;
	cv::threshold(inputImage, binarizedImage, thresholdValue, maxValue, cv::THRESH_BINARY);

	cv::imwrite("Results/binarizedImage_simple_1.jpg", binarizedImage);

	unsigned blockSize = 11, C = 2;
	cv::adaptiveThreshold(inputImage, binarizedImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);

	cv::imwrite("Results/binarizedImage_adaptive_0.jpg", binarizedImage);

	blockSize = 7;
	C = 3;
	cv::adaptiveThreshold(inputImage, binarizedImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);

	cv::imwrite("Results/binarizedImage_adaptive_1.jpg", binarizedImage);

	cv::threshold(inputImage, binarizedImage, 0, maxValue, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::imwrite("Results/binarizedImage_otsu_0.jpg", binarizedImage);

	cv::threshold(inputImage, binarizedImage, 50, maxValue, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::imwrite("Results/binarizedImage_otsu_1.jpg", binarizedImage);

	return 0;
}
