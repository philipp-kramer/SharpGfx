// OpenGlTriangle.cpp : Defines the entry point for the application.
// vcpkg.exe integrate install
// vcpkg.exe install glad
// vcpkg.exe install glfw3 // alternatively generate using https://glad.dav1d.de/
// add e.g. vcpkg\installed\x64-windows\include to VC++ include directories

#include "Shading.h"
#include "Windowing.h"
#include "OpenGL.h"

const char* vertexShaderSource =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";
const char* fragmentShaderSource = 
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

int createTriangle()
{
    float vertices[] = {
        -0.5f, -0.5f, 0.0f, // left  
         0.5f, -0.5f, 0.0f, // right 
         0.0f,  0.5f, 0.0f  // top   
    };

    // bind the Vertex Array Object before binding and setting vertex buffer(s)
    unsigned int vao = genVertexArray();
    glBindVertexArray(vao);

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // configure vertex attributes(s).
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // good practice to always unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return vao;
}

void render(unsigned int shaderProgram, unsigned int object)
{
    glUseProgram(shaderProgram);

    // it is standard procedure to bind and unbind - required with multiple objects
    glBindVertexArray(object);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

int main(void)
{
    GLFWwindow* window = createWindow("Hello World", 800, 600);

    if (window == NULL) return -1;

    unsigned int shaderProgram = compile(vertexShaderSource, fragmentShaderSource);
    unsigned int triangle = createTriangle();
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // uncomment this call to draw in wireframe polygons.
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        glClear(GL_COLOR_BUFFER_BIT);

        render(shaderProgram, triangle);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
