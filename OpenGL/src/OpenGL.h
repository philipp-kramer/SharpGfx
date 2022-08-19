#pragma once
#include <glad/glad.h>
#include "Export.h"

static unsigned int glData[] = { 0 };
static std::int64_t glLongData[] = { 0 };

extern "C"
{
    EXPORT void viewport(const int x, const int y, const int width, const int height) { glViewport(x, y, width, height); }
    EXPORT void clearColor(const float r, const float g, const float b, const float a) { glClearColor(r, g, b, a); }
    EXPORT void clear(const unsigned int buffer_bit) { glClear(buffer_bit); }

    EXPORT unsigned int genVertexArray() { glGenVertexArrays(1, &glData[0]); return glData[0]; }
    EXPORT unsigned int genBuffer() { glGenBuffers(1, &glData[0]); return glData[0]; }
    EXPORT void bindVertexArray(const unsigned int array) { glBindVertexArray(array); }
    EXPORT void bindBuffer(const unsigned int target, const unsigned int buffer) { glBindBuffer(target, buffer); }
    EXPORT void bufferData(const unsigned int target, const long size, const void* data) { glBufferData(target, size, data, GL_STATIC_DRAW); }
    EXPORT std::int64_t getBufferParameter(const unsigned int target, const unsigned int pname) { glGetBufferParameteri64v(target, pname, &glLongData[0]); return glLongData[0];  }
    EXPORT void floatVertexAttribPointer(const unsigned int index, const int size, const int stride, const long long int offset) { glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, stride, (void*)offset); }
    EXPORT void enableVertexAttribArray(const unsigned int array) { glEnableVertexAttribArray(array); }
    EXPORT void drawTriangles(const int first, const int count) { glDrawArrays(GL_TRIANGLES, first, count); }
    EXPORT void drawIndexedTriangles(int count, GLenum type, const void* indices) { glDrawElements(GL_TRIANGLES, count, type, indices); }
    EXPORT void enable(const unsigned int cap) { glEnable(cap); }
    EXPORT void disable(const unsigned int cap) { glDisable(cap); }
    EXPORT void blendFunc(const unsigned int sfactor, const unsigned int dfactor) { glBlendFunc(sfactor, dfactor); }

    EXPORT unsigned int genTexture() { glGenTextures(1, &glData[0]); return glData[0]; }
    EXPORT void bindTexture(const unsigned int target, const unsigned int texture) { glBindTexture(target, texture); }
    EXPORT void texImage2D(const unsigned int target, const int level, const int width, const int height, const int border, const unsigned int format, const unsigned int type, const void* pixels) { glTexImage2D(target, level, format, width, height, border, format, type, pixels); }
    EXPORT void generateMipmap(const unsigned int target) { glGenerateMipmap(target); }
    EXPORT void textureParameterI(const unsigned int target, const unsigned int name, const int parameter) { glTexParameteri(target, name, parameter); }
    EXPORT void activeTexture(const unsigned int texture) { glActiveTexture(texture); }
    EXPORT void deleteTexture(const unsigned int texture) { glData[0] = texture; glDeleteTextures(1, &glData[0]); }

    EXPORT unsigned int genFramebuffer() { glGenFramebuffers(1, &glData[0]); return glData[0]; }
    EXPORT void bindFramebuffer(const unsigned int  target, const unsigned int  framebuffer) { glBindFramebuffer(target, framebuffer); }
    EXPORT void deleteFramebuffer(const unsigned int  framebuffer) { glData[0] = framebuffer; glDeleteFramebuffers(1, &glData[0]); }
    EXPORT void framebufferTexture2D(const unsigned int target, const unsigned int attachment, const unsigned int textarget, const unsigned int texture, int level) { glFramebufferTexture2D(target, attachment, textarget, texture, level); }
    EXPORT unsigned int genRenderbuffer() { glGenRenderbuffers(1, &glData[0]); return glData[0]; }
    EXPORT void bindRenderbuffer(const unsigned int target, const unsigned int renderbuffer) { glBindRenderbuffer(target, renderbuffer); }
    EXPORT void renderbufferStorage(const unsigned int target, const unsigned int internalformat, const int width, const int height) { glRenderbufferStorage(target, internalformat, width, height); }
    EXPORT void framebufferRenderbuffer(const unsigned int target, const unsigned int attachment, const unsigned int renderbuffertarget, const unsigned int renderbuffer) { glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer); }
    EXPORT unsigned int checkFramebufferStatus(const unsigned int target) { return glCheckFramebufferStatus(target); }
    EXPORT void deleteRenderbuffer(const unsigned int renderbuffer) { glData[0] = renderbuffer;  glDeleteRenderbuffers(1, &glData[0]); }
    EXPORT void drawBuffer(const unsigned int buf) { glDrawBuffer(buf); }
    EXPORT void readBuffer(const unsigned int src) { glReadBuffer(src); }

    EXPORT void useProgram(const unsigned int shaderProgram) { glUseProgram(shaderProgram); }
    EXPORT void deleteProgram(const unsigned int shaderProgram) { glDeleteProgram(shaderProgram); }

    EXPORT void deleteVertexArray(const unsigned int array) { glData[0] = array;  glDeleteVertexArrays(1, &glData[0]); }
    EXPORT void deleteBuffer(const unsigned int buffer) { glData[0] = buffer; glDeleteBuffers(1, &glData[0]); }
    EXPORT unsigned int getAttributeLocation(const unsigned int buffer, const char* name) { return glGetAttribLocation(buffer, name); }
    EXPORT unsigned int getUniformLocation(const unsigned int buffer, const char* name) { return glGetUniformLocation(buffer, name); }

    EXPORT void uniform1i(const unsigned int location, const int value) { glUniform1i(location, value); }
    EXPORT void uniform1f(const unsigned int location, const float value) { glUniform1f(location, value); }
    EXPORT void uniform2f(const unsigned int location, const float v1, const float v2) { glUniform2f(location, v1, v2); }
    EXPORT void uniform3f(const unsigned int location, const float v1, const float v2, const float v3) { glUniform3f(location, v1, v2, v3); }
    EXPORT void uniform4f(const unsigned int location, const float v1, const float v2, const float v3, const float v4) { glUniform4f(location, v1, v2, v3, v4); }
    EXPORT void uniformMatrix4f(const unsigned int location, const bool transpose, const float* values) { glUniformMatrix4fv(location, 1, transpose, values); }

    EXPORT void getTexImage(const unsigned int target, const int level, const unsigned int format, const unsigned int type, void* pixels) { glGetTexImage(target, level, format, type, pixels); }
}
