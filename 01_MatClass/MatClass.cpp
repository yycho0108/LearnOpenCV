/*#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
	if(argc != 2){
		std::cout << "Specify Image File!!" << std::endl;
	}
	Mat image;
	image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(! image.data){
		std::cout << "EMPTY IMAGE" << std::endl;
	}

	namedWindow(argv[1],WINDOW_AUTOSIZE);
	imshow(argv[1], image);

	waitKey(0);
	return 0;
}
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
	if(argc != 2){
		std::cout << "SPEICFY FILE " << std::endl;
		return -1;
	}
	Mat img = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(!img.data){
		std::cout << "EMPTY IMAGE" << std::endl;
		return -1;
	}
	namedWindow(argv[1], WINDOW_AUTOSIZE);
	imshow(argv[1],img);
	cvWaitKey(0);
	return 0;
}
