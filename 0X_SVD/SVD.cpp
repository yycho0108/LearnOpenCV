#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <functional>
#include <iostream>

using namespace cv;
using namespace std;
using uchar = unsigned char;

int SIGN(float x){
	return x>0?1:-1;
}

class ForEach : public ParallelLoopBody{
	private:
		float* p;
		function<float(float)> f;
	public:
		ForEach(uchar* ptr, function<float(float)> f):p((float*)ptr),f(f){}
		virtual void operator()(const Range& r) const{
			for(int i= r.start; i != r.end; ++i){ //"register int"?
				//std::cout << "HA!" << std::endl;
				p[i] = f(p[i]);
				if(isnan(p[i]))
					p[i] = 0;
			}
		}
};

void mySVD(Mat& M,Mat& S,Mat& U,Mat& V){
	Mat S1,S2;
	eigen(M*M.t(),S1,U);
	U = U.t(); //column vectors are eigenvectors.
	eigen(M.t()*M,S2,V);
	V = V.t();

	for(int i=0;i<M.rows;++i){
		Mat Check = M.row(0) * V.col(i);
		if( SIGN(Check.at<float>(0,0)/U.at<float>(0,i)) != 1){
			U.col(i) = -U.col(i); // flip if nonnegative
		}
	}

	if (S1.rows < S2.rows){
		S1.copyTo(S);
	}else{
		S2.copyTo(S);
	}

	parallel_for_(Range(0,S.rows*S.cols*S.channels()),ForEach(S.data,[](float a){return sqrt(a);}));
	V = V.t();
}

int k=1;
//Mat img;

uchar dat[5][5] = {{255,20,19,23,128},{40,5,127,200,55},{33,254,44,0,25},{1,4,7,58,2},{90,4,240,1,5}};
Mat img(5,5,CV_8UC1,&dat);

Mat S,U,VT;

void onTrackbarChange(int, void*){
	Mat M = Mat::zeros(img.rows,img.cols,CV_32FC1);
	//std::cout << S1 << std::endl;
	for(int i=10;i<k;++i){
		if(isnan(S.at<float>(0,i)))
			break;
		Mat tmp =  U.col(i) * S.at<float>(i) * VT.row(i);
		cv::add(M,tmp,M);
	}
	parallel_for_(Range(0,M.rows*M.cols*M.channels()),ForEach(M.data,[](float a){return abs(a);}));
	M.convertTo(M,CV_8U);
	
	//std::cout << "M" << endl << M << std::endl;
	//std::cout << endl;
	//std::cout << "IMG" << endl << img << std::endl;

	imshow("SVD",M);
}



int main(int argc, char* argv[]){
	if(argc < 3){
		std::cout << "SPECIFY ALL ARGUMENTS" << std::endl;
		return -1;
	}
	
	img = imread(argv[1],IMREAD_ANYDEPTH);
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	
	//std::cout << "IMG" << img << endl;	
	std::cout << "TYPE : " <<  img.type() << std::endl;	
	img.convertTo(img,CV_32F);
	std::cout << "TYPE : " <<  img.type() << std::endl;	
	//waitKey(0);

	/*Mat MT;
	transpose(img,MT);
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	
	namedWindow("MT",WINDOW_AUTOSIZE);
	imshow("MT",MT);
	waitKey(0);	
	*/
	Mat S1,U1,VT1;
	Mat S2,U2,VT2;

	mySVD(img,S1,U1,VT1);
	cv::SVD::compute(img,S2,U2,VT2,SVD::FULL_UV);
	
	//std::cout << "USUM" << endl << U1+U2 << endl;
	//std::cout << "VSUM" << endl << VT1+VT2 << endl;
	//std::cout << "S1: " << S1 << endl << "U1: " << U1 << endl << "VT1: " <<  VT1 << std::endl << endl;
	//std::cout << "S2: " << S2 << endl << "U2: " << U2 << endl << "VT2: " <<  VT2 << std::endl << endl;
	
	S = S1;
	U = U1;
	VT = VT1;
	k = std::atoi(argv[2]);

	namedWindow("SVD",WINDOW_AUTOSIZE);
	createTrackbar("K","SVD",&k,std::min(img.rows,img.cols),&onTrackbarChange);
	onTrackbarChange(0,nullptr);
	
	//Mat(U2.col(2)).copyTo(U1.col(2));
	//Mat(U2.col(4)).copyTo(U1.col(4));

	namedWindow("VERIFY", WINDOW_AUTOSIZE);
	Mat S_Diag = Mat::zeros(img.rows,img.cols,img.type());
	for(int i=0;i<S1.rows;++i){
		if(!isnan(S1.at<float>(0,i)))
			S_Diag.at<float>(i,i) = S.at<float>(0,i);
	}
	Mat R = U1 * S_Diag * VT1;
	R.convertTo(R,CV_8U);
	//std::cout << "R" << endl << R << endl;
	imshow("VERIFY",R);

	waitKey(0);
	return 0;
}
