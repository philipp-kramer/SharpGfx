#pragma once
#include "glad/glad.h"

static unsigned int glData[] = { 0 };
static long long int glLongData[] = { 0 };

extern "C"
{
    __declspec(dllexport) void viewport(const int x, const int y, const int width, const int height) { glViewport(x, y, width, height); }
    __declspec(dllexport) void clearColor(const float r, const float g, const float b, const float a) { glClearColor(r, g, b, a); }
    __declspec(dllexport) void clear(const unsigned int buffer_bit) { glClear(buffer_bit); }

    __declspec(dllexport) unsigned int genVertexArray() { glGenVertexArrays(1, &glData[0]); return glData[0]; }
    __declspec(dllexport) unsigned int genBuffer() { glGenBuffers(1, &glData[0]); return glData[0]; }
    __declspec(dllexport) void bindVertexArray(const unsigned int array) { glBindVertexArray(array); }
    __declspec(dllexport) void bindBuffer(const unsigned int target, const unsigned int buffer) { glBindBuffer(target, buffer); }
    __declspec(dllexport) void bufferData(const unsigned int target, const long size, const void* data) { glBufferData(target, size, data, GL_STATIC_DRAW); }
    __declspec(dllexport) long long int getBufferParameter(const unsigned int target, const unsigned int pname) { glGetBufferParameteri64v(target, pname, &glLongData[0]); return glLongData[0];  }
    __declspec(dllexport) void floatVertexAttribPointer(const unsigned int index, const int size, const int stride, const long long int offset) { glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, stride, (void*)offset); }
    __declspec(dllexport) void enableVertexAttribArray(const unsigned int array) { glEnableVertexAttribArray(array); }
    __declspec(dllexport) void drawTriangles(const int first, const int count) { glDrawArrays(GL_TRIANGLES, first, count); }
    __declspec(dllexport) void drawIndexedTriangles(int count, GLenum type, const void* indices) { glDrawElements(GL_TRIANGLES, count, type, indices); }
    __declspec(dllexport) void enable(const unsigned int cap) { glEnable(cap); }
    __declspec(dllexport) void disable(const unsigned int cap) { glDisable(cap); }
    __declspec(dllexport) void blendFunc(const unsigned int sfactor, const unsigned int dfactor) { glBlendFunc(sfactor, dfactor); }

    __declspec(dllexport) unsigned int genTexture() { glGenTextures(1, &glData[0]); return glData[0]; }
    __declspec(dllexport) void texImage2D(const unsigned int target, const int level, const int width, const int height, const int border, const unsigned int format, const unsigned int type, const void* pixels) { glTexImage2D(target, level, format, width, height, border, format, type, pixels); }
    __declspec(dllexport) unsigned int createTexture(const unsigned int target) { glCreateTextures(target, 1, glData); return glData[0]; }
    __declspec(dllexport) void textureStorage2D(const unsigned int texture, const int levels, const unsigned int internalformat, const int width, const int height) { glTextureStorage2D(texture, levels, internalformat, width, height); }
    __declspec(dllexport) void textureSubImage2D(const unsigned int texture, const int level, const int xoffset, const int yoffset, const int width, const int height, const unsigned int format, const unsigned int type, const void* pixels) { glTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, type, pixels); }
    __declspec(dllexport) void textureParameterI(const unsigned int texture, const unsigned int pname, const int param) { glTextureParameteri(texture, pname, param); }
    __declspec(dllexport) void activeTexture(const unsigned int texture) { glActiveTexture(texture); }
    __declspec(dllexport) void bindTexture(const unsigned int target, const unsigned int texture) { glBindTexture(target, texture); }
    __declspec(dllexport) void deleteTexture(const unsigned int texture) { glData[0] = texture; glDeleteTextures(1, &glData[0]); }

    __declspec(dllexport) unsigned int genFramebuffer() { glGenFramebuffers(1, &glData[0]); return glData[0]; }
    __declspec(dllexport) void bindFramebuffer(const unsigned int  target, const unsigned int  framebuffer) { glBindFramebuffer(target, framebuffer); }
    __declspec(dllexport) void deleteFramebuffer(const unsigned int  framebuffer) { glData[0] = framebuffer; glDeleteFramebuffers(1, &glData[0]); }
    __declspec(dllexport) void framebufferTexture2D(const unsigned int target, const unsigned int attachment, const unsigned int textarget, const unsigned int texture, int level) { glFramebufferTexture2D(target, attachment, textarget, texture, level); }
    __declspec(dllexport) unsigned int genRenderbuffer() { glGenRenderbuffers(1, &glData[0]); return glData[0]; }
    __declspec(dllexport) void bindRenderbuffer(const unsigned int target, const unsigned int renderbuffer) { glBindRenderbuffer(target, renderbuffer); }
    __declspec(dllexport) void renderbufferStorage(const unsigned int target, const unsigned int internalformat, const int width, const int height) { glRenderbufferStorage(target, internalformat, width, height); }
    __declspec(dllexport) void framebufferRenderbuffer(const unsigned int target, const unsigned int attachment, const unsigned int renderbuffertarget, const unsigned int renderbuffer) { glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer); }
    __declspec(dllexport) unsigned int checkFramebufferStatus(const unsigned int target) { return glCheckFramebufferStatus(target); }
    __declspec(dllexport) void deleteRenderbuffer(const unsigned int renderbuffer) { glData[0] = renderbuffer;  glDeleteRenderbuffers(1, &glData[0]); }
    __declspec(dllexport) void drawBuffer(const unsigned int buf) { glDrawBuffer(buf); }
    __declspec(dllexport) void readBuffer(const unsigned int src) { glReadBuffer(src); }

    __declspec(dllexport) void useProgram(const unsigned int shaderProgram) { glUseProgram(shaderProgram); }
    __declspec(dllexport) void deleteProgram(const unsigned int shaderProgram) { glDeleteProgram(shaderProgram); }

    __declspec(dllexport) void deleteVertexArray(const unsigned int array) { glData[0] = array;  glDeleteVertexArrays(1, &glData[0]); }
    __declspec(dllexport) void deleteBuffer(const unsigned int buffer) { glData[0] = buffer; glDeleteBuffers(1, &glData[0]); }
    __declspec(dllexport) unsigned int getAttributeLocation(const unsigned int buffer, const char* name) { return glGetAttribLocation(buffer, name); }
    __declspec(dllexport) unsigned int getUniformLocation(const unsigned int buffer, const char* name) { return glGetUniformLocation(buffer, name); }

    __declspec(dllexport) void uniform1i(const unsigned int location, const int value) { glUniform1i(location, value); }
    __declspec(dllexport) void uniform1f(const unsigned int location, const float value) { glUniform1f(location, value); }
    __declspec(dllexport) void uniform2f(const unsigned int location, const float v1, const float v2) { glUniform2f(location, v1, v2); }
    __declspec(dllexport) void uniform3f(const unsigned int location, const float v1, const float v2, const float v3) { glUniform3f(location, v1, v2, v3); }
    __declspec(dllexport) void uniform4f(const unsigned int location, const float v1, const float v2, const float v3, const float v4) { glUniform4f(location, v1, v2, v3, v4); }
    __declspec(dllexport) void uniformMatrix4f(const unsigned int location, const bool transpose, const float* values) { glUniformMatrix4fv(location, 1, transpose, values); }

    __declspec(dllexport) void getTexImage(const unsigned int target, const int level, const unsigned int format, const unsigned int type, void* pixels) { glGetTexImage(target, level, format, type, pixels); }
}
