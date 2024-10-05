// Lab1.cpp : Defines the entry point for the application.
//

#include "Lab1.h"
#include <mutex>
#include <thread>

namespace ad {
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

	cv::Mat gammaCorrection(const cv::Mat& const inputImage, double gamma)
	{
		cv::Mat resultImage = inputImage.clone();
		Concurrency::parallel_for(0, inputImage.rows, [&](int i)
			{
				const uchar* inputRow = inputImage.ptr<uchar>(i);
				uchar* resultRow = resultImage.ptr<uchar>(i);
				for (int j = 0; j < inputImage.cols; j++)
					resultRow[j] = std::pow(inputRow[j], gamma);
			});
		return resultImage;
	}

	double MSE(const cv::Mat& const firstImage, const cv::Mat& const secondImage)
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

	double SSIM(const double C1, const double C2, const double C3, const cv::Mat& const firstImage, const cv::Mat& const secondImage, int rowOffset, int colOffset, int blockSize, double alpha, double beta, double gamma)
	{
		double Ux = 0, Uy = 0, sigmaXSqr = 0, sigmaYSqr = 0, sigmaXY = 0, l = 0, c = 0, s = 0;
		const int N = blockSize * blockSize;
		for (int i = rowOffset; i < rowOffset + blockSize; i++)
		{
			const uchar* firstRow = firstImage.ptr<uchar>(i);
			const uchar* secondRow = secondImage.ptr<uchar>(i);
			for (int j = colOffset; j < colOffset + blockSize; j++)
			{
				Ux += firstRow[j];
				Uy += secondRow[j];
			}
		}
		Ux /= N;
		Uy /= N;

		l = (2 * Ux * Uy + C1) / (Ux * Ux + Uy * Uy + C1);

		for (int i = rowOffset; i < rowOffset + blockSize; i++)
		{
			const uchar* firstRow = firstImage.ptr<uchar>(i);
			const uchar* secondRow = secondImage.ptr<uchar>(i);
			for (int j = colOffset; j < colOffset + blockSize; j++)
			{
				sigmaXSqr += (firstRow[j] - Ux) * (firstRow[j] - Ux);
				sigmaYSqr += (secondRow[j] - Uy) * (secondRow[j] - Uy);
				sigmaXY += (firstRow[j] - Ux) * (firstRow[j] - Ux);
			}
		}
		sigmaXSqr /= N;
		sigmaYSqr /= N;
		sigmaXY /= N;
		double sigmaX = sqrt(sigmaXSqr);
		double sigmaY = sqrt(sigmaYSqr);

		c = (2 * sigmaX * sigmaY + C2) / (sigmaXSqr + sigmaYSqr + C2);

		s = (sigmaXY + C3) / (sigmaX * sigmaY + C3);

		return std::pow(l, alpha) * std::pow(c, beta) * std::pow(s, gamma);
	}

	double MS_SSIM(const cv::Mat& const firstImage, const cv::Mat& const secondImage, int blockSize = 16, int step = 8, double alpha = 1, double beta = 1, double gamma = 1)
	{
		const double C1 = std::pow(0.01 * 255, 2);
		const double C2 = std::pow(0.03 * 255, 2);
		const double C3 = C2 / 2;
		std::mutex mtx;
		unsigned int nThreads = std::thread::hardware_concurrency();
		std::vector<double> resultsOfSSIM(nThreads, 0.0);
		Concurrency::parallel_for(0, firstImage.rows - blockSize, step, [&](int i)
			{
				for (int j = 0; j < firstImage.cols - blockSize; j += step)
				{
					int tmpI = i;
					int tmpJ = j;
					if (i + blockSize > firstImage.rows)
						tmpI = firstImage.rows - blockSize;
					if (j + blockSize > firstImage.cols)
						tmpJ = firstImage.cols - blockSize;
					unsigned int threadNum = ((i + 1) / step) % nThreads;
					double valueSSIM = SSIM(C1, C2, C3, firstImage, secondImage, tmpI, tmpJ, blockSize, alpha, beta, gamma);
					std::lock_guard<std::mutex> lock(mtx);
					resultsOfSSIM[threadNum] += valueSSIM;
				}
			});
		double result = 0;
		for (int i = 0; i < nThreads; i++)
			result += resultsOfSSIM[i];
		int blocksPerRow = (firstImage.rows - blockSize) / step;
		int blocksPerCol = (firstImage.cols - blockSize) / step;
		if ((firstImage.rows - blockSize) % step)
			blocksPerRow++;
		if ((firstImage.cols - blockSize) % step)
			blocksPerCol++;
		return result / (blocksPerCol * blocksPerRow);
	}
}

int main()
{
	std::string inputFileName = "E:/HW/ImageProcessing/Lab1/Input/sar_1_gray.jpg";
	cv::Mat inputImage = cv::imread(inputFileName, cv::IMREAD_GRAYSCALE);

	std::filesystem::create_directory("Results");

	cv::Mat histImage = ad::histogramGray(inputImage);
	cv::imwrite("Results/histogram.jpg", histImage);

	cv::Mat gammaCorImage = ad::gammaCorrection(inputImage, 1.1);
	cv::imwrite("Results/gamma.jpg", gammaCorImage);

	std::cout << "MSE = " << ad::MSE(inputImage, gammaCorImage) << std::endl;
	std::cout << "SSIM = " << ad::MS_SSIM(inputImage, gammaCorImage) << std::endl;
}
