#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using uchar = unsigned char;

int main(int argc, char* argv[]){
	if(argc < 2){
		cout << "WRONG # OF ARGS " << endl;
		return -1;
	}
	auto img = imread(argv[1]);
	if(!img.data){
		cout << "EMPTY IMAGE" << endl;
		return -1;
	}

	Mat res(img.rows,img.cols,img.type());

	auto nRows = img.rows;
	auto nCols = img.cols;
	auto nChannels = img.channels();

	// "SHARPEN"
	for(int i=1;i<nRows;++i){
		uchar* prev = img.ptr<uchar>(i-1);
		uchar* cur = img.ptr<uchar>(i);
		uchar* next = img.ptr<uchar>(i+1);
		
		uchar* r = res.ptr<uchar>(i);
		for(int j=nChannels;j<nCols*nChannels;++j){
			r[j] = saturate_cast<uchar>(5*cur[j] - (cur[j-nChannels] + cur[j+nChannels] + prev[j] + next[j]));
		}
	}
	// "SHARPEN With Filters"
	Mat res_2(nRows,nCols,img.type());
	Mat ker = (Mat_<char>(3,3) << 0,-1,0,
		-1,5,-1,
		0,-1,0);

	filter2D(img,res_2,img.depth(),ker);
	namedWindow("ORIGINAL",WINDOW_AUTOSIZE);
	imshow("ORIGINAL",img);
	
	namedWindow("MASK",WINDOW_AUTOSIZE);
	imshow("MASK",res);

	namedWindow("MASK_F2D",WINDOW_AUTOSIZE);
	imshow("MASK_F2D",res_2);
	
	waitKey(0);
}
