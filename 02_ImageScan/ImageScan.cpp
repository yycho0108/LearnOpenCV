#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

using uchar = unsigned char;

void makeTable(uchar* table, int div){
	for(int i=0;i<256;++i){
		//table[i] = (uchar) i/div * div;
		table[i] = 255 - i;
		std::cout << (int)table[i] << ' ';
	}
	std::cout << std::endl;
}
Mat& ScanAndReduceImage(Mat& img,const uchar* const table){
	int nRows = img.rows;
	int nCols = img.cols * img.channels();
	for(int i=0;i<nRows;++i){
		uchar* p = img.ptr<uchar>(i);
		for(int j=0;j<nCols;++j){
			p[j] = table[p[j]];
		}
	}
	return img;
}

int main(int argc, char* argv[]){
	if(argc < 3){
		std::cout << " Specify file and division factor" << std::endl;
		return -1;
	}
	//uchar table[256];
	int div = atoi(argv[2]);
	//makeTable(table, div);
	auto table = Mat(1,256,CV_8U);
	makeTable(table.data, div);

	Mat img = imread(argv[1],IMREAD_UNCHANGED);
	if(!img.data){
		std::cout << "EMPTY IMAGE" << std::endl;
		return -1;
	}

	//ScanAndReduceImage(img,table);
	LUT(img,table,img);
	namedWindow(argv[1],WINDOW_AUTOSIZE);
	imshow(argv[1],img);
	waitKey(0);
	
}
