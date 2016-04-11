#ifndef __POOLLAYER_H__
#define __POOLLAYER_H__
#include "Layer.h"

//currently only supports ave-pooling by opencv
class PoolLayer : public Layer{
	Size s_i;
	Size s_p,s_s; //pooling size, stride size
	std::vector<std::vector<std::vector<Point>>> S; //switches
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;

public:
	PoolLayer(Size i, Size o);

	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);

	virtual Size outputSize();
	virtual void setup(Size);
	virtual void save(std::string dir);
};

#endif
