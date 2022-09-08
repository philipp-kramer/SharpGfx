#pragma once
using namespace std; 
#include <iostream>
#include <string>
#include <limits.h>
#include <math.h>
#include <GLFW/glfw3.h>
#include "Export.h"

static unsigned int last_key = UINT_MAX;
static unsigned int last_button = UINT_MAX;
static unsigned int last_action = UINT_MAX;
static unsigned int new_width = UINT_MAX;
static unsigned int new_height = UINT_MAX;

static double last_pos_x = NAN;
static double last_pos_y = NAN;
static double scroll_x = 0;
static double scroll_y = 0;

void error_callback(int error, const char* msg) {
	std::cerr << " [ OpenGL Error " << std::to_string(error) << ": " << msg << " ] " << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    new_width = width;
    new_height = height;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
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
    if (!glfwInit()) {
        std::cout << "Failed to initialize glfw" << std::endl;
        return NULL;
    }
    std::cout << "SharpGfx.OpenGL version 1.2.4" << std::endl;

    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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
    EXPORT void* createGlfWindow(const char* title, const int width, const int height) { return (void*) createWindow(title, width, height); }
    EXPORT bool isWindowCloseRequested(void* glfWindow) { glfwPollEvents(); return glfwWindowShouldClose((GLFWwindow*) glfWindow); }
    EXPORT void swapBuffers(void* glfWindow) { glfwSwapBuffers((GLFWwindow*)glfWindow); }
    EXPORT void terminateGlfw() { glfwTerminate(); }

    EXPORT unsigned int getNewWidth() { unsigned int width = new_width; new_width = UINT_MAX; return width; }
    EXPORT unsigned int getNewHeight() { unsigned int heigth = new_height; new_height = UINT_MAX; return heigth; }
    EXPORT unsigned int getKey() { unsigned int key = last_key; last_key = UINT_MAX; return key; }
    EXPORT unsigned int getMouseButton() { unsigned int button = last_button; last_button = UINT_MAX; return button; }
    EXPORT unsigned int getMouseAction() { unsigned int action = last_action; last_action = UINT_MAX; return action; }

    EXPORT double getMousePosX() { return last_pos_x; }
    EXPORT double getMousePosY() { return last_pos_y; }
    EXPORT double getMouseScrollX() { double x = scroll_x; scroll_x = 0; return x; }
    EXPORT double getMouseScrollY() { double y = scroll_y; scroll_y = 0; return y; }
}
