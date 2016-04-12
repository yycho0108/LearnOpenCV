#ifndef __CONVOLUTIONLAYER_H__
#define __CONVOLUTIONLAYER_H__
#include "Layer.h"

class ConvolutionLayer : public Layer{
private:
	Size s;
	int d_i, d_o; //depth of input layers, depth of output layers
	float m; //momentum
	bool** connection;
	
	std::vector<Mat> W;
	std::vector<Mat> b;
	std::vector<Mat> dW;
	std::vector<Mat> dW_p; //previous
	std::vector<Mat> db;
	std::vector<Mat> db_p; //previous

	std::vector<Mat> g; //local gain matrix

	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;

public:
	ConvolutionLayer(int,int);
	~ConvolutionLayer();

	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	virtual void update();

	virtual Size outputSize();
	virtual void setup(Size);
	std::vector<Mat>& getW();

	virtual void save(FileStorage& f, int i);
	virtual void load(FileStorage& f, int i);
};

#endif
