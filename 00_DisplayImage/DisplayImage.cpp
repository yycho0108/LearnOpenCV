#include <iostream>
#include <opencv/cv.hpp>

using namespace cv;

int main(int argc, char** argv){
	if(argc != 2){
		std::cout <<"** Specify Image File! **";
		return -1;
	}

	Mat image;
	image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(!image.data){
		std::cout << " NO ImAge DATA";
		return -1;
	}


	Mat gray_image;
	cvtColor(image,gray_image,CV_BGR2GRAY);

	namedWindow("DISPLAY IMAGE", WINDOW_AUTOSIZE);
	imshow("DISPLAY IMAGE",gray_image);
	waitKey(0);
	return 0;
}
