#include "window.h"
#include "Exception.h"


static void errorCallback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}


static void windowSizeCallback(GLFWwindow* glfWindow, int32_t res_x, int32_t res_y)
{
    Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfWindow));

    // Keep rendering at the current resolution when the window is minimized.
    if (window->minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    ensureMinimumSize(res_x, res_y);

    window->width = res_x;
    window->height = res_y;
    window->resizeDirty = true;
}


static void windowIconifyCallback(GLFWwindow* glfWindow, int32_t iconified)
{
    Window* properties = static_cast<Window*>(glfwGetWindowUserPointer(glfWindow));
    properties->minimized = (iconified > 0);
}


static void keyCallback(GLFWwindow* glfWindow, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(glfWindow, true);
        }
        else
        {
            Window* properties = static_cast<Window*>(glfwGetWindowUserPointer(glfWindow));
            properties->inputs.key = key;
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}


static void cursor_position_callback(GLFWwindow* glfWindow, double x, double y)
{
    Window* properties = static_cast<Window*>(glfwGetWindowUserPointer(glfWindow));
    properties->inputs.mouseX = x;
    properties->inputs.mouseY = y;
}

static void mouse_button_callback(GLFWwindow* glfWindow, int button, int action, int mods)
{
    Window* properties = static_cast<Window*>(glfwGetWindowUserPointer(glfWindow));
    properties->inputs.button = button;
    properties->inputs.action = action;
}

static void mouse_scroll_callback(GLFWwindow* glfWindow, double x, double y)
{
    Window* properties = static_cast<Window*>(glfwGetWindowUserPointer(glfWindow));
    properties->inputs.scrollX = x;
    properties->inputs.scrollY = y;
}

void initGL()
{
    if (!gladLoadGL())
        throw std::runtime_error("Failed to initialize GL");

    GL_CHECK(glClearColor(0.212f, 0.271f, 0.31f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}


GLFWwindow* initGLFW(const char* window_title, int width, int height)
{
    GLFWwindow* glfWindow = nullptr;
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        throw Exception("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // To make Apple happy -- should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfWindow = glfwCreateWindow(width, height, window_title, nullptr, nullptr);
    if (!glfWindow)
        throw Exception("Failed to create GLFW window");

    glfwMakeContextCurrent(glfWindow);
    glfwSwapInterval(0);  // No vsync

    return glfWindow;
}


GLFWwindow* Window::initGlWindow(const char* title, int width, int height)
{
    GLFWwindow* glfWindow = initGLFW(title, width, height);
    glfwSetWindowSizeCallback(glfWindow, windowSizeCallback);
    glfwSetWindowIconifyCallback(glfWindow, windowIconifyCallback);
    glfwSetKeyCallback(glfWindow, keyCallback);
    glfwSetCursorPosCallback(glfWindow, cursor_position_callback);
    glfwSetMouseButtonCallback(glfWindow, mouse_button_callback);
    glfwSetScrollCallback(glfWindow, mouse_scroll_callback);
    glfwSetWindowUserPointer(glfWindow, this);

    initGL();
    gl_display = new GLDisplay();
    return glfWindow;
}


Inputs get_inputs(Window& window)
{ 
    return window.get_inputs(); 
}

bool window_should_close(Window& window) 
{ 
    return window.windowShouldClose(); 
}
