#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <functional>
#include <iostream>
#include <fstream>
#include <ctime>

#include <signal.h>

#include "ConvNet.h"
#include "Parser.h"

using namespace cv;
using namespace std;


static volatile bool keepTraining = true;
static volatile bool keepTesting = true;

void intHandler(int){
	if(keepTraining){
		keepTraining = false;
	}else{
		keepTesting = false;
	}
}

void setup(ConvNet& net){
	/* ** CONV LAYER TEST ** */

	net.push_back(new ConvolutionLayer(1,12));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));

	net.push_back(new ConvolutionLayer(12,16));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
		
	net.push_back(new FlattenLayer(16));
	net.push_back(new DenseLayer(1,84));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new DenseLayer(1,10));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new SoftMaxLayer());

	net.setup(Size(28,28));
}

void visualize(ConvNet& net){ //doesn't really work (since by nature convolution is translation invariant)
	std::vector<Mat> X({Mat::zeros(28,28,DataType<float>::type)});
	std::vector<Mat> Active;
	Active.reserve(10);
	for(int i=0;i<10;++i){
		//ensure each matrix in different
		Active.push_back(Mat::zeros(28,28,DataType<float>::type));
	}
	for(int i=0;i<28;++i){
		for(int j=0;j<28;++j){
			X[0].at<float>(i,j) = 1.0;//activate
			auto Y = net.FF(X);
			for(int k=0;k<10;++k){
				Active[k].at<float>(i,j) = Y[0].at<float>(0,k); //or 10,1?
			}
			X[0].at<float>(i,j) = 0.0;
		}
	}

	Mat im(Size(100,100), DataType<float>::type);
	for(int i=0;i<10;++i){
		auto s = "A" +  std::to_string(i);
		namedWindow(s,WINDOW_AUTOSIZE);
		//double maxVal = 0;
		//cv::minMaxLoc(Active[i],nullptr,&maxVal);
		//Active[i] /= maxVal;
		cv::resize(Active[i],im,im.size());
		//std::cout << im << std::endl;
		imshow(s,im);
	}
	waitKey();
}

void train(ConvNet& net, int lim){
	std::cout << "TRAINING FOR : " << lim << std::endl;
	keepTraining = true;

	Parser trainer("../data/trainData","../data/trainLabel");
	std::vector<Mat> X(1),Y(1);

	int i = 0;

	while (1){
		if(!trainer.read(X[0],Y[0])){//EOF
			trainer.reset();
		}

		if(++i > lim || !keepTraining)
			return;

		if(!(i%100)){
			cout << "TRAINING ... " << i << endl;
		}
		auto Yp = net.FF(X);
		//cout << "YP: " << Yp[0].t() << endl;
		//cout << "YL " << Y[0].t() << endl;
		net.BP(Yp,Y);
	}

	keepTraining = false;
}

void test(ConvNet& net){
	keepTesting = true;
	
	Parser tester("../data/testData","../data/testLabel");

	Mat d,l;
	std::vector<Mat> X(1),Y(1);

	int cor=0;
	int inc = 0;

/* VISUALIZING THE LEARNED KERNELS */	
	//const auto& L = net.getL();
	//
	//tester.read(X[0],Y[0]);
	//tester.reset();//reset immediately to not affect the later testing
	//
	////auto& K = ((ConvolutionLayer*)L[0])->getW();
	//
	//namedWindow("X",WINDOW_AUTOSIZE);
	//imshow("X",X[0]);

	//auto& K = L[0]->FF(X);

	//Mat im(Size(100,100), DataType<float>::type);
	//for(size_t i=0;i<K.size();++i){
	//	auto s = "K" +  std::to_string(i);
	//	auto& k = K[i];
	//	//cout << k << endl;
	//	cv::resize(k,im,im.size());
	//	
	//	namedWindow(s,WINDOW_AUTOSIZE);
	//	imshow(s,im);
	//}

	//waitKey();
/* END */

	while(tester.read(X[0],Y[0]) && keepTesting){ //read into X,Y
		auto y0 = net.FF(X)[0];
		cout << "OUTPUT : " << endl << y0.t() << endl;
		//cout << "TARGET : " << endl <<  Y[0].t() << endl;
		auto y = argmax(y0);
		auto t = argmax(Y[0]);
		y==t?(++cor):(++inc);
		cout << "O[" << argmax(net.FF(X)[0]) << "]:T[" << argmax(Y[0]) <<"]"<<endl;
		printf("%d cor, %d inc\n", cor,inc);
	}

	keepTesting = false;
}
void parseParams(int argc, char* argv[], int& lim){
	enum{NONE=-1};
	char opt = NONE;

	for(int i=0;i<argc;++i){
		switch(opt){
			case 'd':
				DECAY = std::atof(argv[i]);
				opt = NONE;
				break;
			case 'e':
				ETA = std::atof(argv[i]);
				opt = NONE;
				break;
			case 'l':
				lim = std::atoi(argv[i]);
				opt = NONE;
				break;
			default:
				break;
		}
		if(argv[i][0] == '-'){
			opt = std::tolower(argv[i][1]);
		}

	}
}

int main(int argc, char* argv[]){
	/* Testing 3d Matrix Creation/Dereferencing
	 * const int sz[] = {1,2,3};
	Mat testing(3,sz,DataType<float>::type);
	for(int i=0;i<testing.dims;++i){
		cout << testing.size[i] << ',';
	}*/

	signal(SIGINT, intHandler);

	int lim = 60000;
	parseParams(argc,argv,lim);

	ConvNet net;

	setup(net);
	train(net, lim);
	//visualize(net);
	test(net);


	net.save("save"); //save to directory

	return 0;

}
