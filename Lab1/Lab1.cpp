// Lab1.cpp : Defines the entry point for the application.
//

#include "Lab1.h"
#include <mutex>
#include <thread>

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

	/*int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;
	cv::Mat histogram;
	cv::calcHist(&inputImage, 1, 0, cv::Mat(), histogram, 1, &histSize, histRange, uniform, accumulate);
	double histMathExpectation = 0;
	double histStandardDeviation = 0;
	for (int i = 0; i < histSize; i++)
		histMathExpectation += i * histogram.ptr<float>(0)[i];
	histMathExpectation /= inputImage.rows * inputImage.cols;
	for (int i = 0; i < histSize; i++)
		histStandardDeviation += (i - histMathExpectation) * (i - histMathExpectation) * histogram.ptr<float>(0)[i];
	histStandardDeviation = std::sqrt(histStandardDeviation / (inputImage.rows * inputImage.cols));
	cv::Mat statisticallyCorrectedImage = imgPro.statisticalColorCorrection(histMathExpectation, histStandardDeviation, inputImage);*/

	cv::Mat equalizedImage;
	cv::equalizeHist(inputImage, equalizedImage);
	cv::Mat statisticallyCorrectedImage = imgPro.statisticalColorCorrection(equalizedImage, inputImage);
	cv::imwrite("Results/statisticallyCorrectedImage.jpg", statisticallyCorrectedImage);

	std::cout << "Statistically corrected image MSE = " << imgPro.MSE(inputImage, statisticallyCorrectedImage) << std::endl;
	std::cout << "Statistically corrected image SSIM = " << imgPro.MS_SSIM(inputImage, statisticallyCorrectedImage) << std::endl;
}
