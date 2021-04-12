#pragma once

#include <iostream>
#include "Logger.h"
#include "OSDefines.h"
using namespace std;


class LoggerToConsole : public Logger {
public:
	void print(string message)
	{
		cout << message;
	}

	void println(string message)
	{
		cout << message << endl;
	}

private:

};
