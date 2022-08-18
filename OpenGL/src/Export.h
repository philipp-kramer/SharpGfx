#pragma once

#if defined(_MSC_VER)
//  Microsoft 
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#pragma warning Unknown dynamic link export semantics.
#endif