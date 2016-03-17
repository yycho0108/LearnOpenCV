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
	x = sigmoid(x);
	return x * (1-x);
}

void sigmoid(Mat& src, Mat& dst){
	if(&dst != &src){
		src.copyTo(dst);
	}
	ForEach(dst.data,[](float x){return sigmoid(x);});
}

void sigmoidPrime(Mat& src, Mat& dst){
	if(&dst != &src){
		src.copyTo(dst);
	}
	ForEach(dst.data,[](float x){return sigmoidPrime(x);});
}

void convolve(cv::InputArray I, cv::OutputArray O,cv::InputArray W){
	return cv::filter2D(I,O,-1,W,Point(-1,-1),0.0,BORDER_CONSTANT);//change depth later
}
Mat max_pool(Mat& m, Size s){
	Mat res;
	cv::pyrDown(m,res,s,cv::BORDER_REPLICATE);
	return res;
}


/* ** Layer Base Class ** */

class Layer{
	public:
		virtual std::vector<Mat>& FF(std::vector<Mat>)=0;
		virtual std::vector<Mat>& BP(std::vector<Mat>)=0;
};

/* ** Activation Layer ** */
class ActivationLayer : public Layer{
private:
	int d;
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G; //maybe not necessary? idk...
public:
	ActivationLayer(int d);
	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
};

ActivationLayer::ActivationLayer(int d):d(d),I(d),O(d){

}
std::vector<Mat>& ActivationLayer::FF(std::vector<Mat> _I){
	//assert same size
	I.swap(_I);
	for(int i=0;i<d;++i){
		sigmoid(I[i],O[i]);
	}
	return O;
}
std::vector<Mat>& ActivationLayer::BP(std::vector<Mat> _G){
	Mat tmp;
	for(int i=0;i<d;++i){
		sigmoidPrime(I[i],tmp);
		_G[i].mul(tmp);
	}
	G.swap(_G);
	return G;
}

/* ** Convolution Layer ** */

class ConvLayer : public Layer{
private:
	int d_i, d_o; //depth of input layers, depth of output layers
	bool** connection;
	
	std::vector<Mat> W;
	std::vector<Mat> b;
	std::vector<Mat> dW;

	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;

public:
	ConvLayer(int,int,Size);
	~ConvLayer();
	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	void update();
};
ConvLayer::ConvLayer(int d_i, int d_o, Size s)
	:d_i(d_i),d_o(d_o),dW(d_o){
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
		cv::randn(W[o],cv::Scalar::all(0),cv::Scalar::all(0.1));

		b.push_back(Mat(s,DataType<float>::type,Scalar::all(0.1)));//wrong dimension though!	
		O.push_back(Mat(s,DataType<float>::type));//wrong dimension though!	
	}
}
ConvLayer::~ConvLayer(){

	for(int i=0;i<d_i;++i){
		delete connection[i];
	}
	delete[] connection;

}
std::vector<Mat>& ConvLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	G = std::vector<Mat>(I.size());

	for(int i=0;i<d_i;++i){
		for(int o=0;o<d_o;++o){
			if(connection[i][o]){
				cout << i << ',' <<  o << endl;
				convolve(I[i],O[o],W[o]);
				cv::filter2D(I[i],O[o],I[i].depth(),W[o]);//change depth later
				//and maybe even replace this function with something less rigid.
			}
		}
	}
	for(int o=0;o<d_o;++o){
		//O[o] /= 3.0;
		O[o] += b[o];
		//sigmoid(O[o],O[o]); //in-place sigmoid
		//subsample
		//auto w = O[o].size().width/2;
		//auto h = O[o].size().height/2;
		//O[o] = max_pool(O[o],cv::Size(w,h));
	}
	return O;
}

std::vector<Mat>& ConvLayer::BP(std::vector<Mat> _G){
	//_G.size() == d_o
	//G.size() == d_i
	
	auto iw = I[0].size().width;
	auto ih = I[0].size().height;
	
	auto ow = _G[0].size().width;
	auto oh = _G[0].size().height;

	auto fwr = W[0].size().width/2; //kernel size
	auto fhr = W[0].size().height/2;
	
	for(int i=0;i<d_i;++i){
		G[i] = Mat::zeros(I[i].size(),DataType<float>::type);
	}

	for(int o=0;o<d_o;++o){ //for each output channel(depth):
		sigmoidPrime(O[o],O[o]);
		_G[o] = _G[o].mul(O[o]); //element-wise
		dW[o] = Mat::zeros(O[o].size(),DataType<float>::type);//initialize dW

		for(int i=0;i<d_i;++i){ //for each input channel
			//G[i].setTo(Scalar(0)); //set all elements to zero
			if(connection[i][o]){ //if the channels are related.. 
				for(int y=0; y<ih;++y){
					for(int x=0;x<iw;++x){

						auto ymin = max(y-fhr,0);
						auto ymax = min(y+fhr+1,oh);//assume odd kernel
						auto xmin = max(x-fwr,0);
						auto xmax = min(x+fwr+1,ow);
						
						G[i](cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin)) += 
							W[o](cv::Rect(xmin-x+fwr,ymin-y+fhr,xmax-xmin,ymax-ymin))
						   	* _G[o].at<float>(y,x);
						dW[o](cv::Rect(xmin-x+fwr,ymin-y+fhr,xmax-xmin,ymax-ymin)) += I[i](cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin)) * _G[o].at<float>(y,x);
						//may not be right index
					}
				}
			}
		}

	}

	return G;
}
void ConvLayer::update(){
	for(int o=0;o<d_o;++o){
		W[o] += dW[o];
	}
}
/* ** Pooling Layer ** */
class PoolLayer{
	Size i,o;
public:
	std::vector<Mat>& FF(std::vector<Mat> I);
	std::vector<Mat>& BP(std::vector<Mat> G);
	PoolLayer(Size i, Size o);
};

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
	
	auto cl = ConvLayer(1,3,img.size());
	auto al = ActivationLayer(3);

	std::vector<Mat> I;
	I.push_back(img);
	
	namedWindow("K1",WINDOW_AUTOSIZE);
	namedWindow("K2",WINDOW_AUTOSIZE);
	namedWindow("K3",WINDOW_AUTOSIZE);

	auto m = cl.FF(I);
	cl.BP(m);
	m = al.FF(m);

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
	testConvLayer(argc,argv);
	/*if(argc != 2){
		cout << "SPECIFY CORRECT ARGS" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);

*/
	return 0;
}
