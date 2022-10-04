# How to build this

Install glfw and then run the following:

```
g++ \
  -fdiagnostics-color=always \
  -g \
  OpenGL.cpp \
  glad.c \
  -o \
  OpenGL.dll \
  -I \
  include \
  -I \
  /usr/include/GLFW \
  -lglfw \
  -ldl \
  -shared \
  -fPIC
```

