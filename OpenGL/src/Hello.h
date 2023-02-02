#include "Export.h"
#include <iostream>

extern "C"
{
    EXPORT void console() { std::cout << "hello c++" << std::endl; }
}