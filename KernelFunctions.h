#pragma once

#include "OSDefines.h"
class KernelFunctor {
public:
	virtual double eval(double d) = 0;
};

class EpanechnikovKernel : public KernelFunctor {
	double eval(double d) override {
		return (3.0 / 4.0) * (1.0 - d * d);
	}
};