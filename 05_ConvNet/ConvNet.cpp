#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>
#include <iostream>

using namespace cv;
using namespace std;

/* ** UTILITY ** */


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

float sigmoid(float x){
	return 1.0/(1.0 + exp(-x));
}

float sigmoidPrime(float x){
	return 1.0/(1.0 + exp(-x));
}

void sigmoid(Mat& src, Mat& dst){
	if(&dst != &src){
		src.copyTo(dst);
	}
	ForEach(dst.data,[](float x){return sigmoid(x);});
}

Mat max_pool(Mat& m, Size s){
	Mat res;
	cv::pyrDown(m,res,s,cv::BORDER_REPLICATE);
	return res;
}

/* ** ConvLayer ** */

class ConvLayer{
private:
	int d_i, d_o; //depth of input layers, depth of output layers
	bool** connection;
	
	std::vector<Mat> W;
	std::vector<Mat> b;

	std::vector<Mat> _I;
	std::vector<Mat> _O;
	std::vector<Mat> _G;

public:
	ConvLayer(int,int,Size);
	~ConvLayer();
	std::vector<Mat>& transfer(std::vector<Mat> I);
};
ConvLayer::ConvLayer(int d_i, int d_o, Size s)
	:d_i(d_i),d_o(d_o){
	// Size Before SubSampling
	// d_i = depth of input layers
	// d_o = depth of output layers
	connection = new bool*[d_i];
	for(int i=0;i<d_i;++i){
		connection[i] = new bool[d_o];
		for(int o=0;o<d_o;++o){
			connection[i][o] = true;
		}
	}

	for(int o=0;o<d_o;++o){

		W.push_back(Mat(5,5,DataType<float>::type));
		cv::randu(W[o],cv::Scalar::all(0),cv::Scalar::all(0.3));

		b.push_back(Mat(s,DataType<float>::type,Scalar::all(0.1)));//wrong dimension though!	
		_O.push_back(Mat(s,DataType<float>::type));//wrong dimension though!	
	}
}
ConvLayer::~ConvLayer(){

	for(int i=0;i<d_i;++i){
		delete connection[i];
	}
	delete[] connection;

}
std::vector<Mat>& ConvLayer::transfer(std::vector<Mat> I){
	_I.swap(I);

	for(int i=0;i<d_i;++i){
		for(int o=0;o<d_o;++o){
			if(connection[i][o]){
				cout << i << ',' <<  o << endl;
				cv::filter2D(_I[i],_O[o],_I[i].depth(),W[o]);//change depth later
			}
		}
	}
	for(int o=0;o<d_o;++o){
		_O[o] += b[o];
		sigmoid(_O[o],_O[o]); //in-place sigmoid
		//subsample
		auto w = _O[o].size().width/2;
		auto h = _O[o].size().height/2;
		_O[o] = max_pool(_O[o],cv::Size(w,h));
	}
	return _O;
}

class ConvNet{

};


int testConvLayer(int argc, char* argv[]){
	if(argc != 2){
		cout << "SPECIFY IMG FILE" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);
	
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	
	img.convertTo(img,CV_32F);
	auto l = ConvLayer(1,3,img.size());
	std::vector<Mat> I;
	I.push_back(img);
	
	namedWindow("K1",WINDOW_AUTOSIZE);
	namedWindow("K2",WINDOW_AUTOSIZE);
	namedWindow("K3",WINDOW_AUTOSIZE);

	auto m = l.transfer(I);
	m[0].convertTo(m[0],CV_8U);
	m[1].convertTo(m[1],CV_8U);
	m[2].convertTo(m[2],CV_8U);

	imshow("K1", m[0]);
	imshow("K2", m[1]);
	imshow("K3", m[2]);
	waitKey();
	return 0;
}

int main(int argc, char* argv[]){
	if(argc != 2){
		cout << "SPECIFY CORRECT ARGS" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);


	return 0;
}
