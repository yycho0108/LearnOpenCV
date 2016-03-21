#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(int d_i, int d_o)
	:d_i(d_i),d_o(d_o),dW(d_o){
	// Size Before SubSampling
	// d_i = depth of input layers
	// d_o = depth of output layers
	connection = new bool*[d_o];
	//often o>i
	for(int o=0;o<d_o;++o){
		connection[o] = new bool[d_i];
		for(int i=0;i<d_i;++i){
			connection[o][i] = true;//((o%3) != (i%3));
			/*if(o%3 != i%3){ // ~2/3 connection
				connection[i][o] = true;
			}*/
		}
	}

	for(int o=0;o<d_o;++o){
		W.push_back(Mat(5,5,DataType<float>::type));//hard-coded kernel size
		cv::randn(W[o],cv::Scalar::all(0),cv::Scalar::all(0.1));
	}
}
ConvolutionLayer::~ConvolutionLayer(){

	for(int i=0;i<d_i;++i){
		delete connection[i];
	}
	delete[] connection;

}
std::vector<Mat>& ConvolutionLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	G = std::vector<Mat>(I.size());
	//cout << "W[0] : " << endl << W[0] << endl;

	for(int o=0;o<d_o;++o){
		O[o] = Mat::zeros(O[o].size(),O[o].type());
		//cout << "W[o] : " << endl << W[o] << endl;
		for(int i=0;i<d_i;++i){
			if(connection[o][i]){
				//cout << i << ',' <<  o << endl;
				Mat tmp;
				correlate(I[i],tmp,W[o],true); //true convolution
				O[o] += tmp;
				//cv::filter2D(I[i],O[o],I[i].depth(),W[o]);//change depth later
				//and maybe even replace this function with something less rigid.
			}
		}
		O[o] += b[o];
		//if(isnan(O[o])){
		//	throw "OISNAN";
		//}
	}
	return O;
}

std::vector<Mat>& ConvolutionLayer::BP(std::vector<Mat> _G){
	//_G.size() == d_o
	//G.size() == d_i
	
	auto iw = s.width;
	auto ih = s.height;
	//auto ow = s.width;
	//auto oh = s.height;

	//auto iw = I[0].size().width;
	//auto ih = I[0].size().height;
	
	//auto ow = _G[0].size().width;
	//auto oh = _G[0].size().height;
	
	auto fw = W[0].size().width; //kernel size
	auto fh = W[0].size().height;
	auto fwr = fw/2; //kernel size
	auto fhr = fh/2;

	//cout << _G[0] << endl;	

	for(int i=0;i<d_i;++i){
		G[i] = Mat::zeros(I[i].size(),DataType<float>::type);
	}

	for(int o=0;o<d_o;++o){ //for each output channel(depth):
		//sigmoidPrime(O[o],O[o]);
		//_G[o] = _G[o].mul(O[o]); //element-wise --> don't need since activation layer is separate now
		//
		dW[o] = Mat::zeros(W[o].size(),DataType<float>::type);//initialize dW
		//db[o] = Mat::zeros(b[o].size(),DataType<float>::type);

		//Mat K;
		//flip(W[o],K,-1);
		for(int i=0;i<d_i;++i){ //for each input channel
			
			if(connection[o][i]){ //if the channels are related.. 

				Mat tmp;
				correlate(_G[o],tmp,W[o],false); //correlation (convolution with flipped kernel)
				G[i] += tmp;

				//correlate(_G[o],tmp,I[i],false); //correlation
				//dW[o] += tmp;
				
				for(int y=0;y<fh;++y){
					for(int x=0;x<fw;++x){
						//I[0,0] maps to G[x,y]
						
						Mat&& I_dw = I[i](cv::Rect(max(0,fwr-x),max(0,fhr-y),iw-abs(x-fwr),ih-abs(y-fhr)));
						Mat&& G_dw = _G[o](cv::Rect(max(0,x-fwr),max(0,y-fhr),iw-abs(x-fwr),ih-abs(y-fhr)));
						
						//printf("X:%d,Y:%d\n",x-fwr,y-fhr);
						//printf("I : (%d,%d) [%d,%d]\n", max(0,fwr-x),max(0,fhr-y),iw-abs(x-fwr),ih-abs(y-fhr));
						//printf("G : (%d,%d) [%d,%d]\n", max(0,x-fwr),max(0,y-fhr),iw-abs(x-fwr),ih-abs(y-fhr));
						
						dW[o].at<float>(y,x) += cv::sum(I_dw.mul(G_dw))[0];  

						//cout << "ADD : " << v  << endl;
					}
				}	
				/*for(int y=0; y<ih;++y){
					for(int x=0;x<iw;++x){

						auto ymin = max(y-fhr,0);
						auto ymax = min(y+fhr+1,oh);//assume odd kernel
						auto xmin = max(x-fwr,0);
						auto xmax = min(x+fwr+1,ow);
						
						//G[i](cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin)) += 
						//	K(cv::Rect(xmin-x+fwr,ymin-y+fhr,xmax-xmin,ymax-ymin))
						//   	* _G[o].at<float>(y,x);
						auto val = _G[o].at<float>(y,x);
						dW[o](cv::Rect(xmin-x+fwr,ymin-y+fhr,xmax-xmin,ymax-ymin)) += I[i](cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin)) * val;
						
						//db[o] += I[i] * _G[o].at<float>(y,x);
						//may not be right index
					}
				}*/
			}
		}
		//cv::flip(dW[o],dW[o],-1);
		dW[o] -= W[o]*DECAY;
		db[o] = _G[o]; //bias = gradient
	}

	return G;
}
void ConvolutionLayer::update(){
	for(int o=0;o<d_o;++o){
		W[o] += ETA * dW[o];
		b[o] += ETA * db[o];
	}
}

void ConvolutionLayer::setup(Size s){
	this->s = s;
	b.clear();
	O.clear();
	db.clear();
	for(int o=0;o<d_o;++o){
		db.push_back(Mat(s,DataType<float>::type,Scalar::all(0.0)));//wrong dimension though!	
		b.push_back(Mat(s,DataType<float>::type,Scalar::all(0.1)));//wrong dimension though!	
		O.push_back(Mat(s,DataType<float>::type));//wrong dimension though!	
	}
}

Size ConvolutionLayer::outputSize(){
	return s; //same since conv=same
}
