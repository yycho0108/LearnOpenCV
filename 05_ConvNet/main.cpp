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

void train(ConvNet& net, int lim){
	keepTraining = true;

	Parser trainer("../data/trainData","../data/trainLabel");
	std::vector<Mat> X(1),Y(1);

	int i = 0;
	int max_epoch = 1;
	for(int epoch=0;epoch<max_epoch;++epoch){
		while (trainer.read(X[0],Y[0])){

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
		trainer.reset();
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
	const auto& L = net.getL();
	
	
	tester.read(X[0],Y[0]);
	tester.reset();//reset immediately to not affect the later testing
	
	//auto& K = ((ConvolutionLayer*)L[0])->getW();
	
	namedWindow("X",WINDOW_AUTOSIZE);
	imshow("X",X[0]);

	auto& K = L[0]->FF(X);

	Mat im(Size(100,100), DataType<float>::type);
	for(size_t i=0;i<K.size();++i){
		auto s = "K" +  std::to_string(i);
		auto& k = K[i];
		cout << k << endl;
		cv::resize(k,im,im.size());
		
		namedWindow(s,WINDOW_AUTOSIZE);
		imshow(s,im);
	}

	waitKey();
/* END */

	while(tester.read(X[0],Y[0]) && keepTesting){ //read into X,Y
		cout << "OUTPUT : " << endl << net.FF(X)[0].t() << endl;
		//cout << "TARGET : " << endl <<  Y[0].t() << endl;
		auto y = argmax(net.FF(X)[0]);
		auto t = argmax(Y[0]);
		y==t?(++cor):(++inc);
		cout << "O[" << argmax(net.FF(X)[0]) << "]:T[" << argmax(Y[0]) <<"]"<<endl;
		printf("%d cor, %d inc\n", cor,inc);

	}

	keepTesting = false;
}

int main(int argc, char* argv[]){
	signal(SIGINT, intHandler);

	int lim = 60000;

	if(argc != 1){
		lim = std::atoi(argv[1]);
	}
	cout << lim << endl;

	ConvNet net;

	setup(net);
	train(net, lim);
	test(net);

	return 0;

}
