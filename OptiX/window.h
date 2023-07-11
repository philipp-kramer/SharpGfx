#pragma once

#include "buffering.h"
#include "CUDAOutputBuffer.h"
#include "GLDisplay.h"
#include "context.h"

#include <GLFW/glfw3.h>


struct Inputs
{
    int key = 0;
    int button = 0;
    int action = 0;
    double mouseX = 0;
    double mouseY = 0;
    double scrollX = 0;
    double scrollY = 0;
};

class Window
{
public:
    unsigned int width;
    unsigned int height;
    bool resizeDirty;
    bool minimized;
    Inputs inputs;

    Window(const char* title, unsigned int width, unsigned int height)  :
        glfWindow(initGlWindow(title, width, height)), 
        width(width), 
        height(height), 
        resizeDirty(false), 
        minimized(false),
        inputs({})
    {
    }

    Inputs get_inputs()
    {
        Inputs result = inputs;
        inputs.key = 0;
        inputs.scrollX = 0;
        inputs.scrollY = 0;
        return result;
    }

    void displaySubframe(GLuint pbo)
    {
        // Display
        int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
        int framebuf_res_y = 0;  //
        glfwGetFramebufferSize(glfWindow, &framebuf_res_x, &framebuf_res_y);
        gl_display->display(width, height, framebuf_res_x, framebuf_res_y, pbo );

        glfwSwapBuffers(glfWindow);
    }

    bool windowShouldClose()
    {
        return glfwWindowShouldClose(glfWindow);
    }

    ~Window()
    {
        delete gl_display;
        glfwDestroyWindow(glfWindow);
        glfwTerminate();
    }

private:
    GLDisplay* gl_display;
    GLFWwindow* glfWindow;

    GLFWwindow* initGlWindow(const char* title, int width, int height);
};

extern "C"
{
    __declspec(dllexport) Inputs get_inputs(Window& window);
    __declspec(dllexport) bool window_should_close(Window& window);
}