#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>

using namespace cv;

Mat dropout(Mat& I, double p){
	Mat mask(I.size(),cv::DataType<float>::type);
	cv::randu(mask,0.0,1.0);
	cv::threshold(mask,mask,p,1.0,cv::THRESH_BINARY);
	return I.mul(mask);
}

int main(){
	cv::theRNG().state = time(0);
	Mat I(5,5,cv::DataType<float>::type);
	cv::randu(I,0,1.0);

	std::cout << I << std::endl;
	std::cout << dropout(I,0.5) << std::endl;

}
