#pragma once
using namespace std; 
#include <GLFW/glfw3.h>
#include <iostream>

static unsigned int last_key = UINT_MAX;
static unsigned int last_button = UINT_MAX;
static unsigned int last_action = UINT_MAX;
static unsigned int width_change = UINT_MAX;
static unsigned int height_change = UINT_MAX;

static double last_pos_x = NAN;
static double last_pos_y = NAN;
static double scroll_x = 0;
static double scroll_y = 0;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    width_change = width;
    height_change = height;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        last_key = key;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    switch (button)
    {
    case GLFW_MOUSE_BUTTON_LEFT:
    case GLFW_MOUSE_BUTTON_MIDDLE:
    case GLFW_MOUSE_BUTTON_RIGHT:
        last_button = button;
        break;
    }
    switch (action)
    {
    case GLFW_PRESS:
    case GLFW_RELEASE:
        last_action = action;
        break;
    }
}

void mouse_callback(GLFWwindow* window, double pos_x_d, double pos_y_d)
{
    last_pos_x = pos_x_d;
    last_pos_y = pos_y_d;
}

void scroll_callback(GLFWwindow* window, double offset_x, double offset_y)
{
    scroll_x += offset_x;
    scroll_y += offset_y;
}

GLFWwindow* createWindow(const char* title, const int width, const int height)
{
    if (!glfwInit()) return NULL;

    GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window)
    {
        std::cout << "Failed to create window" << std::endl;
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // tell GLFW to capture our mouse
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return NULL;
    }

    glfwSetKeyCallback(window, keyCallback);

    return window;
}

extern "C"
{
    __declspec(dllexport) void* createGlfWindow(const char* title, const int width, const int height) { return (void*) createWindow(title, width, height); }
    __declspec(dllexport) bool isWindowCloseRequested(void* glfWindow) { glfwPollEvents(); return glfwWindowShouldClose((GLFWwindow*) glfWindow); }
    __declspec(dllexport) void swapBuffers(void* glfWindow) { glfwSwapBuffers((GLFWwindow*)glfWindow); }
    __declspec(dllexport) void terminateGlfw() { glfwTerminate(); }

    __declspec(dllexport) void getEvents(unsigned int* data) 
    {
        data[0] = last_key;
        data[1] = last_button;
        data[2] = last_action;
        data[3] = width_change;
        data[4] = height_change;
        last_key = UINT_MAX;
        last_button = UINT_MAX;
        last_action = UINT_MAX;
        width_change = UINT_MAX;
        height_change = UINT_MAX;
    }

    __declspec(dllexport) void getMouseInputs(double* data)
    { 
        data[0] = last_pos_x; 
        data[1] = last_pos_y; 
        data[2] = scroll_x; 
        data[3] = scroll_y; 
        scroll_x = 0;
        scroll_y = 0;
    }
}
