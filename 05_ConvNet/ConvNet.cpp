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
	//currently zero-padding
}
Mat ave_pool(Mat& m, Size s){
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
/* ** Dense Layer ** */

class DenseLayer : public Layer{
private:
	int d,s_i,s_o;
	std::vector<Mat> W;
	std::vector<Mat> b;
	std::vector<Mat> dW;
	std::vector<Mat> db;
	
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G; //maybe not necessary? idk...
	//
public:
	DenseLayer(int d, int i, int o);
	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	void update();
};

DenseLayer::DenseLayer(int d,int s_i,int s_o)
	:d(d),s_i(s_i),s_o(s_o),W(d),b(d),dW(d),db(d),I(d),O(d),G(d){
	for(int i=0;i<d;++i){
		W[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		b[i] = Mat::zeros(Size(1,s_o),DataType<float>::type);
		dW[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		db[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		cv::randn(W[i],cv::Scalar::all(0),cv::Scalar::all(0.1));
	}
}

std::vector<Mat>& DenseLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	for(size_t i=0;i<I.size();++i){
		O[i] = W[i]*I[i]+b[i];
	}
	return O;
}
std::vector<Mat>& DenseLayer::BP(std::vector<Mat> _G){
	for(int i=0;i<d;++i){
		G[i] = W[i].t() * _G[i];
		dW[i] = 0.6 * _G[i]*I[i].t(); //bit iffy in here, but I guess... since no sigmoid.
		db[i] = 0.6 * _G[i];
	}
	return G;
}
void DenseLayer::update(){
	for(int i=0;i<d;++i){
		W[i] += dW[i];
		b[i] += db[i];
	}	
}

class FlattenLayer : public Layer{
private:
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;
	Size s;
public:
	FlattenLayer(Size s);
	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
};

FlattenLayer::FlattenLayer(Size s):s(s){
	O.push_back(Mat());
}

std::vector<Mat>& FlattenLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	int n = I.size();
	s = I[0].size();
	O[0] = I[0].reshape(0,s.width*s.height);
	for(int i=1;i<n;++i){
		cv::vconcat(O[0],I[i].reshape(0,s.width*s.height),O[0]);
	}
	return O;
}

std::vector<Mat>& FlattenLayer::BP(std::vector<Mat> _G){
	G.resize(I.size());
	int n = _G[0].size().height;

	int l = s.width*s.height;

	for(int i=0;i<n/l;++i){
		G[i] = Mat(_G[0](cv::Rect(0,i*l,1,l)).reshape(0,s.height));
	}
	return G;
}

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
	Size s;
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
	:s(s),d_i(d_i),d_o(d_o),dW(d_o){
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

		W.push_back(Mat(5,5,DataType<float>::type));//hard-coded kernel size
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
				//cv::filter2D(I[i],O[o],I[i].depth(),W[o]);//change depth later
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
		//O[o] = ave_pool(O[o],cv::Size(w,h));
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
//currently only supports ave-pooling by opencv
class PoolLayer : public Layer{
	Size s_p,s_s; //pooling size, stride size
	std::vector<std::vector<std::vector<Point>>> S; //switches
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;

public:
	std::vector<Mat>& FF(std::vector<Mat> I);
	std::vector<Mat>& BP(std::vector<Mat> G);
	PoolLayer(Size i, Size o);
};

PoolLayer::PoolLayer(Size s_p, Size s_s):s_p(s_p),s_s(s_s){

}

std::vector<Mat>& PoolLayer::FF(std::vector<Mat> _I){
	S.resize(_I.size());
	O.resize(_I.size());

	I.swap(_I);
	auto pw = s_p.width;
	auto ph = s_p.height;
	auto sw = s_s.width;
	auto sh = s_s.height;
	auto iw = I[0].size().width;
	auto ih = I[0].size().height;
	auto it_w = (iw - pw + sw-1) / sw;
	auto it_h = (ih - ph + sh-1) / sh;


	double maxVal;
	int maxIdx[2];

	for(size_t i=0;i<I.size();++i){
		S[i].resize(it_h);
		O[i] = Mat(it_h,it_w,DataType<float>::type);
		for(int _y=0;_y<it_h;++_y){
			S[i][_y].resize(it_w);
			for(int _x=0;_x<it_w;++_x){

				auto y = _y*sh;
				auto x = _x*sw;

				if(y+ph >= ih || x+pw >= iw){
					auto _ph = min(ph,ih-y);
					auto _pw = min(pw,iw-x);
					cv::minMaxIdx(I[i](Rect(x,y,_pw,_ph)),nullptr,&maxVal,nullptr,maxIdx);
				}else{
					cv::minMaxIdx(I[i](Rect(x,y,pw,ph)),nullptr,&maxVal,nullptr,maxIdx);
				}
				S[i][_y][_x] = Point(maxIdx[1],maxIdx[0]);
				O[i].at<float>(_y,_x) = maxVal;

			}

		}
	}
	
	return O;
}
std::vector<Mat>& PoolLayer::BP(std::vector<Mat> _G){
	G.resize(_G.size()); // = resize depth (channel)

	int h = S[0].size();
	int w = S[0][0].size();

	auto sw = s_s.width;
	auto sh = s_s.height;

	for(size_t i=0;i<_G.size();++i){
		G[i] = Mat::zeros(I[i].size(),DataType<float>::type);
		for(int _y=0;_y<h;++_y){
			for(int _x=0;_x<w;++_x){
				auto& loc = S[i][_y][_x];
				G[i].at<float>(_y*sh +loc.y, _x*sw + loc.x) = _G[i].at<float>(_y,_x);
			}
		}
	}
	return G;
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

int testDenseLayer(int argc, char* argv[]){
	if(argc != 2){
		cout << "SPECIFY IMG FILE" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);
	
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	img.convertTo(img,CV_32F);
	auto dl = DenseLayer(1, img.rows*img.cols, 3);
	auto fl = FlattenLayer(img.size());

	std::vector<Mat> I;
	I.push_back(img);

	auto m = fl.FF(I);
	m = dl.FF(m);
	dl.BP(m);

	m[0].convertTo(m[0],CV_8U);
	//m[1].convertTo(m[1],CV_8U);
	//m[2].convertTo(m[2],CV_8U);

	imshow("M0", m[0]);
	//imshow("M1", m[1]);
	//imshow("M2", m[2]);
	waitKey();
	return 0;
}

int testPoolLayer(int argc, char* argv[]){
	if(argc != 2){
		cout << "SPECIFY IMG FILE" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);
	
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	img.convertTo(img,CV_32F);
	auto pl = PoolLayer(Size(5,5),Size(2,2));
	std::vector<Mat> I;
	I.push_back(img);

	auto m = pl.FF(I);
	pl.BP(m);
	m[0].convertTo(m[0],CV_8U);

	imshow("M", m[0]);
	waitKey();
	return 0;
}


int main(int argc, char* argv[]){
	//testPoolLayer(argc,argv);
	//testDenseLayer(argc,argv);
	
	if(argc != 2){
		cout << "SPECIFY CORRECT ARGS" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);

	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	img.convertTo(img,CV_32F);

	std::vector<Mat> I;
	I.push_back(img);

	auto cl_1 = ConvLayer(1,6,img.size());
	auto al_1 = ActivationLayer(6);
	auto pl_1 = PoolLayer(Size(5,5),Size(3,3));
	auto cl_2 = ConvLayer(6,16,Size(127,90));
	auto al_2 = ActivationLayer(16);
	auto pl_2 = PoolLayer(Size(5,5),Size(3,3));
	auto fl = FlattenLayer(Size(3,3)); // arbitrary... frankly don't know
	auto dl = DenseLayer(1,19024,10);
	auto al_3 = ActivationLayer(1);

	auto m = cl_1.FF(I);
	m = al_1.FF(m);
	m = pl_1.FF(m);
	m = cl_2.FF(m);
	m = al_2.FF(m);
	m = pl_2.FF(m);
	m = fl.FF(m);
	m = dl.FF(m);
	m = al_3.FF(m);

	std::cout << m[0] << endl;

	m = al_3.BP(m);
	m = dl.BP(m);
	m = fl.BP(m);
	m = pl_2.BP(m);
	m = al_2.BP(m);
	// confirmed working until here
	m = cl_2.BP(m);
	m = pl_1.BP(m);
	m = cl_1.BP(m);

	std::cout << m[0] << endl;



	return 0;
}
