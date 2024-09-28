// Lab1.cpp : Defines the entry point for the application.
//

#include "Lab1.h"

cv::Mat histogramGray(const cv::Mat& inputImage)
{
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;
	cv::Mat histogram;
	cv::calcHist(&inputImage, 1, 0, cv::Mat(), histogram, 1, &histSize, histRange, uniform, accumulate);
	int hist_w = 1920, hist_h = 1080;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}
	return histImage;
}

cv::Mat gammaCorrection(const cv::Mat& inputImage, double gamma)
{
	cv::Mat resultImage = inputImage.clone();
	Concurrency::parallel_for(0, inputImage.rows, [&](int i)
		{
			for (int j = 0; j < inputImage.cols; j++)
				resultImage.at<uchar>(i, j) = std::pow(inputImage.at<uchar>(i, j), gamma);
		});
	return resultImage;
}

double calcMSE(const cv::Mat& firstImage, const cv::Mat& secondImage)
{
	double tmp = 0.0;
	int rows = firstImage.rows;
	int cols = firstImage.cols;
	for (int i = 0; i < rows; i++)
	{
		const uchar* firstRow = firstImage.ptr<uchar>(i);
		const uchar* secondRow = secondImage.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
			tmp += (firstRow[j] - secondRow[j]) * (firstRow[j] - secondRow[j]);
	}
	return tmp / (rows * cols);
}

int main()
{
	std::string inputFileName = "E:/HW/ImageProcessing/Lab1/Input/sar_1_gray.jpg";
	cv::Mat inputImage = cv::imread(inputFileName, cv::IMREAD_GRAYSCALE);

	std::filesystem::create_directory("Results");

	cv::Mat histImage = histogramGray(inputImage);
	cv::imwrite("Results/histogram.jpg", histImage);

	cv::Mat gammaCorImage = gammaCorrection(inputImage, 1.1);
	cv::imwrite("Results/gamma.jpg", gammaCorImage);

	std::cout << "MSE = " << calcMSE(inputImage, gammaCorImage) << std::endl;
}
