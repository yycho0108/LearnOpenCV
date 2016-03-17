#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
using namespace cv;
using namespace std;
using uchar = unsigned char;


Mat img1;
Mat img2;
Mat res;
int a=128;

void onChange(int,void*){
	auto alpha = a / 256.0;
	addWeighted(img1,alpha,img2,1.0-alpha,0.0,res);
	imshow("Lerp",res);
}

int main(int argc, char* argv[]){
	if(argc != 3){
		cout << "WRONG # ARGS" << endl;
		return -1;
	}
	img1 = imread(argv[1]);
	img2 = imread(argv[2]);
	if(!img1.data || !img2.data){
		cout << "EMPTY IMAGE" << endl;
		return -1;
	}

	if(img1.type() != img2.type()){
		img2.convertTo(img2,img1.type());
	}
	
	if(img1.size() != img2.size()){
		resize(img2,img2,img1.size());
	}

	namedWindow("Lerp",WINDOW_AUTOSIZE);
	createTrackbar("alpha","Lerp",&a,256,&onChange);
	onChange(0,nullptr);
	waitKey(0);

	return 0;
}
