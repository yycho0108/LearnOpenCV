#include "ConvNet.h"
#include "Parser.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

int testConvolutionLayer(int argc, char* argv[]){
	if(argc != 2){
		cout << "SPECIFY IMG FILE" << endl;
		return -1;
	}
	auto img = imread(argv[1],IMREAD_ANYDEPTH);
	
	cv::theRNG().state = time(0);
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);
	img.convertTo(img,CV_32F,1/256.0);
	
	auto cl = ConvolutionLayer(1,4);
	//auto al = ActivationLayer("sigmoid");
	
	cl.setup(img.size());
	//al.setup(cl.outputSize());


	std::vector<Mat> I;
	I.push_back(img);
	

	auto m = cl.FF(I);
	cl.BP(m);
	//m = al.FF(m);
	for(int i=0;i<4;++i){
		namedWindow(std::to_string(i),WINDOW_AUTOSIZE);
		m[i].convertTo(m[i],CV_8U,256.0);
		imshow(std::to_string(i),m[i]);
	}

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
	auto dl = DenseLayer(1, 3);
	auto fl = FlattenLayer(1);

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
	auto pl = PoolLayer(Size(2,2),Size(2,2));
	std::vector<Mat> I;
	I.push_back(img);

	auto m = pl.FF(I);
	pl.BP(m);
	m[0].convertTo(m[0],CV_8U);

	imshow("M", m[0]);
	waitKey();
	return 0;
}

int testLayerStack(int argc, char* argv[]){
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

	auto cl_1 = ConvolutionLayer(1,6);
	auto al_1 = ActivationLayer("ReLU");
	auto pl_1 = PoolLayer(Size(5,5),Size(3,3));
	auto cl_2 = ConvolutionLayer(6,16);
	auto al_2 = ActivationLayer("ReLU");
	auto pl_2 = PoolLayer(Size(5,5),Size(3,3));
	auto fl = FlattenLayer(16); // arbitrary... frankly don't know
	auto dl = DenseLayer(1,10);
	auto al_3 = ActivationLayer("sigmoid");

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

int testConvNet(int argc, char* argv[]){
	if(argc != 2){
		cout << "SPECIFY CORRECT ARGS" << endl;
		cout << argv[0] << endl;
		return -1;
	}
	/*auto img = imread(argv[1],IMREAD_ANYDEPTH);
	namedWindow("M",WINDOW_AUTOSIZE);
	imshow("M",img);*/

	Mat img = (Mat_<float>(3,3) << 1,2,3,4,5,6,7,8,9);
	img.convertTo(img,CV_32F,1/256.0);

	Mat cla = Mat::zeros(10,1,DataType<float>::type);
	cla.at<float>(4,0) = 1.0;

	std::vector<Mat> X;
	X.push_back(img);
	std::vector<Mat> Y;
	Y.push_back(cla);
	
	ConvNet net;
	net.push_back(new ConvolutionLayer(1,3));
	net.push_back(new ActivationLayer("softplus"));
	//net.push_back(new PoolLayer(Size(5,5),Size(3,3)));
	//net.push_back(new ConvolutionLayer(6,16));
	//net.push_back(new ActivationLayer("ReLU"));
	//net.push_back(new PoolLayer(Size(5,5),Size(3,3)));
	net.push_back(new FlattenLayer(3));
	net.push_back(new DenseLayer(1,10));
	//net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new SoftMaxLayer());
	net.setup(img.size());

	auto m = net.FF(X);
	
	std::cout << "M" << endl << m[0] << endl;

	for(int i=0;i<100;++i){
		auto Yp = net.FF(X);
		cout << Yp[0].t() << endl;
		net.BP(Yp,Y);
	}

	std::cout << "_M_" << endl << m[0] << endl;
	m = net.FF(X);
	std::cout << "M" << endl << m[0] << endl;
	std::cout << "TARGET " << endl << Y[0] << endl;

	//auto m = net.FF(I);
	//std::cout << "M" << endl << m[0] << endl;

	////hope that BP will change weights
	//std::vector<Mat> n(1);
	//m[0].copyTo(n[0]);
	//parallel_for_(Range(0,n[0].rows*n[0].cols),ForEach(n[0].data,[](float a){return a - 0.1;}));

	//std::cout << "N" << endl << n[0] << endl;
	//
	//net.BP(I,n);

	//m = net.FF(I);
	//std::cout << "M_NOW: " << endl << m[0] << endl;

	return 0;
}

int testMNIST(int argc, char* argv[]){

	int lim = 60000;
	if(argc != 1){
		lim = std::atoi(argv[1]);
	}
	ConvNet net;

	/* ** DENSE LAYER TEST ** */
	//net.push_back(new FlattenLayer(1));
	//net.push_back(new DenseLayer(1,75));
	//net.push_back(new ActivationLayer("sigmoid"));
	//net.push_back(new DenseLayer(1,10));
	//net.push_back(new ActivationLayer("sigmoid"));
	//net.push_back(new SoftMaxLayer());
	
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
	
	/* ** POOL LAYER TEST ** */
	//net.push_back(new PoolLayer(Size(2,2),Size(2,2)));
	//net.push_back(new FlattenLayer(1));
	//net.push_back(new DenseLayer(1,75));
	//net.push_back(new ActivationLayer("sigmoid"));
	//net.push_back(new DenseLayer(1,10));
	//net.push_back(new ActivationLayer("sigmoid"));
	//net.push_back(new SoftMaxLayer());

	net.setup(Size(28,28));
	
	Parser trainer("../data/trainData","../data/trainLabel");
	Mat d,l;
	std::vector<Mat> X(1),Y(1);
	int i=0;
	int max_epoch=1;

	for(int epoch=0;epoch<max_epoch;++epoch){
		while (trainer.read(d,l)){
			//cout << d << endl;
			++i;
			if(!(i%100)){
				cout << i << endl;
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


	Parser tester("../data/trainData","../data/trainLabel");

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
	return 0;
}
