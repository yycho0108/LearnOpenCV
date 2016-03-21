#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <functional>
#include <iostream>
#include <fstream>
#include <ctime>

#include "ConvNet.h"
#include "Parser.h"

using namespace cv;
using namespace std;

void setup(ConvNet& net){
	/* ** CONV LAYER TEST ** */
	net.push_back(new ConvolutionLayer(1,3));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));

	//net.push_back(new ConvolutionLayer(2,1));
	//net.push_back(new ActivationLayer("relu"));
	//net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
	
	net.push_back(new FlattenLayer(3));
	net.push_back(new DenseLayer(1,84));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new DenseLayer(1,10));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new SoftMaxLayer());

	net.setup(Size(28,28));
}

void train(ConvNet& net, int max_epoch){

	Parser trainer("../data/trainData","../data/trainLabel");
	Mat d,l;
	std::vector<Mat> X(1),Y(1);

	for(int epoch=0;epoch<max_epoch;++epoch){
		while (trainer.read(d,l)){
			//cout << d << endl;
			if(!(epoch%100)){
				cout << epoch << endl;
			}
			X[0] = d;
			Y[0] = l;
			auto Yp = net.FF(X);
			//cout << "YP: " << Yp[0].t() << endl;
			//cout << "YL " << Y[0].t() << endl;
			net.BP(Yp,Y);
		}
		trainer.reset();
	}

}

void test(ConvNet& net){
	
	Parser tester("../data/trainData","../data/trainLabel");

	Mat d,l;
	std::vector<Mat> X(1),Y(1);

	int cor=0;
	int inc = 0;

/* VISUALIZING THE LEARNED KERNELS */	
	/*namedWindow("M",WINDOW_AUTOSIZE);
	trainer.read(d,l);
	X[0] = d;
	imshow("M",X[0]);
	const auto& L = net.getL();
	auto& K = L[0]->FF(X);
	namedWindow("K",WINDOW_AUTOSIZE);
	imshow("K",K[0]);
	waitKey();*/
/* END */

	while(tester.read(d,l)){
		X[0] = d;
		Y[0] = l;
		cout << "OUTPUT : " << endl << net.FF(X)[0].t() << endl;
		//cout << "TARGET : " << endl <<  Y[0].t() << endl;
		auto y = argmax(net.FF(X)[0]);
		auto t = argmax(Y[0]);
		y==t?(++cor):(++inc);
		cout << "O[" << argmax(net.FF(X)[0]) << "]:T[" << argmax(Y[0]) <<"]"<<endl;
		printf("%d cor, %d inc\n", cor,inc);

	}

}

int main(int argc, char* argv[]){

	int lim = 60000;

	if(argc != 1){
		lim = std::atoi(argv[1]);
	}

	ConvNet net;

	setup(net);
	train(net, lim);
	test(net);

	return 0;

}
