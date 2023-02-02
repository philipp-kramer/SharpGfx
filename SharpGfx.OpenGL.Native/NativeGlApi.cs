using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpGfx.Primitives;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx.OpenGL.Native;

public class NativeGlApi : GlApi
{
    internal NativeGlApi() {}

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "viewport", CallingConvention = CallingConvention.Cdecl)]
    private static extern void viewport(int x, int y, int width, int height);
    protected override void Viewport(int x, int y, int width, int height) { viewport(x, y, width, height); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "clearColor", CallingConvention = CallingConvention.Cdecl)]
    private static extern void clearColor(float r, float g, float b, float a);
    protected override void ClearColor(float r, float g, float b, float a) { clearColor(r, g, b, a); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "clear", CallingConvention = CallingConvention.Cdecl)]
    private static extern void clear(GlBufferBit bufferBit);
    protected override void Clear(GlBufferBit bufferBit) { clear(bufferBit); }


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "genVertexArray", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint genVertexArray();
    protected override uint GenVertexArray() { return genVertexArray(); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "genBuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint genBuffer();
    protected override uint GenBuffer() { return genBuffer(); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "bufferData", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void bufferData(GlBufferTarget target, long size, void* data);
    protected override unsafe void BufferData(GlBufferTarget target, int size, IntPtr data) { bufferData(target, size, data.ToPointer()); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getBufferParameter", CallingConvention = CallingConvention.Cdecl)]
    private static extern long getBufferParameter(GlBufferTarget target, GlBufferParameterName name);
    protected override long GetBufferParameter(GlBufferTarget target, GlBufferParameterName name) { return getBufferParameter(target, name); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "floatVertexAttribPointer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void floatVertexAttribPointer(uint index, int size, int stride, int offset);
    protected override void FloatVertexAttribPointer(uint index, int size, int stride, int offset) { floatVertexAttribPointer(index, size, stride * sizeof(float), offset * sizeof(float)); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindVertexArray", CallingConvention = CallingConvention.Cdecl)]
    private static extern void bindVertexArray(uint array);
    protected override void BindVertexArray(uint array) { bindVertexArray(array); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindBuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void bindBuffer(GlBufferTarget target, uint buffer);
    protected override void BindBuffer(GlBufferTarget target, uint buffer) { bindBuffer(target, buffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "enableVertexAttribArray", CallingConvention = CallingConvention.Cdecl)]
    private static extern void enableVertexAttribArray(uint array);
    protected override void EnableVertexAttribArray(uint array) { enableVertexAttribArray(array); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "drawTriangles", CallingConvention = CallingConvention.Cdecl)]
    private static extern void drawTriangles(int first, int count);
    protected override void DrawTriangles(int count) { drawTriangles(0, count); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "drawIndexedTriangles", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void drawIndexedTriangles(int count, GlType type, void* indices);
    private static GlType GetElementsType<T>()
    {
        if (typeof(T) == typeof(byte))
        {
            return GlType.UByte;
        }
        else if (typeof(T) == typeof(ushort))
        {
            return GlType.UShort;
        }
        else if (typeof(T) == typeof(uint))
        {
            return GlType.UInt;
        }
        else
        {
            throw new NotSupportedException($"type {typeof(T)} not supported for indexing");
        }
    }
    protected override unsafe void DrawIndexedTriangles<T>(int count, IntPtr indices) { drawIndexedTriangles(count, GetElementsType<T>(), indices.ToPointer()); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "enable", CallingConvention = CallingConvention.Cdecl)]
    private static extern void enable(GlCap cap);
    protected override void Enable(GlCap cap) { enable(cap); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "disable", CallingConvention = CallingConvention.Cdecl)]
    private static extern void disable(GlCap cap);
    protected override void Disable(GlCap cap) { disable(cap); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "blendFunc", CallingConvention = CallingConvention.Cdecl)]
    private static extern void blendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor);
    protected override void BlendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor) { blendFunc(srcFactor, dstFactor); }


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "genTexture", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint genTexture();
    protected override uint GenTexture() { return genTexture(); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindTexture", CallingConvention = CallingConvention.Cdecl)]
    private static extern void bindTexture(GlTextureTarget target, uint texture);
    protected override void BindTexture(GlTextureTarget target, uint texture) { bindTexture(target, texture); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "texImage2D", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void texImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, void* pixels);

    protected override unsafe void TexImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, IntPtr pixels) { texImage2D(target, level, width, height, border, format, type, pixels.ToPointer()); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "generateMipmap", CallingConvention = CallingConvention.Cdecl)]
    private static extern void generateMipmap(GlTextureTarget target);
    protected override void GenerateMipmap(GlTextureTarget target) { generateMipmap(target); }
        
    [DllImport(@"x64/OpenGL.dll", EntryPoint = "textureParameterI", CallingConvention = CallingConvention.Cdecl)]
    private static extern void textureParameterI(GlTextureTarget target, GlTextureParameterName name, int parameter);
    protected override void TextureParameterI(GlTextureTarget target, GlTextureParameterName name, GlTextureParameter parameter) { textureParameterI(target, name, (int) parameter); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "activeTexture", CallingConvention = CallingConvention.Cdecl)]
    private static extern void activeTexture(GlTextureUnit glTexture);
    protected override void ActiveTexture(GlTextureUnit unit) { activeTexture(unit); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteTexture", CallingConvention = CallingConvention.Cdecl)]
    private static extern void deleteTexture(uint texture);
    protected override void DeleteTexture(uint texture) { deleteTexture(texture); }


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "genFramebuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint genFramebuffer();
    protected override uint GenFramebuffer() { return genFramebuffer(); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindFramebuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void bindFramebuffer(GlFramebufferTarget target, uint framebuffer);
    protected override void BindFramebuffer(GlFramebufferTarget target, uint framebuffer) { bindFramebuffer(target, framebuffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteFramebuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void deleteFramebuffer(uint framebuffer);
    protected override void DeleteFramebuffer(uint framebuffer) { deleteFramebuffer(framebuffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "framebufferTexture2D", CallingConvention = CallingConvention.Cdecl)]
    private static extern void framebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget textarget, uint texture, int level);
    protected override void FramebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget texTarget, uint texture, int level) { framebufferTexture2D(target, attachment, texTarget, texture, level); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "genRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint genRenderbuffer();
    protected override uint GenRenderbuffer() { return genRenderbuffer(); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "bindRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void bindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer);
    protected override void BindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer) { bindRenderbuffer(target, renderbuffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "renderbufferStorage", CallingConvention = CallingConvention.Cdecl)]
    private static extern void renderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalformat, int width, int height);
    protected override void RenderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalFormat, int width, int height) { renderbufferStorage(target, internalFormat, width, height); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "framebufferRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void framebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbuffertarget, uint renderbuffer);
    protected override void FramebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbufferTarget, uint renderbuffer) { framebufferRenderbuffer(target, attachment, renderbufferTarget, renderbuffer); }
        
    [DllImport(@"x64/OpenGL.dll", EntryPoint = "checkFramebufferStatus", CallingConvention = CallingConvention.Cdecl)]
    private static extern GlFramebufferErrorCode checkFramebufferStatus(GlFramebufferTarget target);
    protected override GlFramebufferErrorCode CheckFramebufferStatus(GlFramebufferTarget target) { return checkFramebufferStatus(target); }
        
    [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteRenderbuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void deleteRenderbuffer(uint renderbuffer);
    protected override void DeleteRenderbuffer(uint renderbuffer) { deleteRenderbuffer(renderbuffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "drawBuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void drawBuffer(uint buf);
    protected override void DrawBuffer(GlDrawBufferMode buffer) { drawBuffer((uint) buffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "readBuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void readBuffer(uint src);
    protected override void ReadBuffer(GlReadBufferMode src) { readBuffer((uint) src); }

        
    [DllImport(@"x64/OpenGL.dll", EntryPoint = "compile", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint compile(string vertexShaderSource, string fragmentShaderSource);
    protected override uint Compile(string vertexShaderSource, string fragmentShaderSource, string fragColorChannel, List<string> errors) { return compile(vertexShaderSource, fragmentShaderSource); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "useProgram", CallingConvention = CallingConvention.Cdecl)]
    private static extern void useProgram(uint shaderProgram);
    protected override void UseProgram(uint shaderProgram) { useProgram(shaderProgram); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteProgram", CallingConvention = CallingConvention.Cdecl)]
    public static extern void deleteProgram(uint handle);
    protected override void DeleteProgram(uint handle) { deleteProgram(handle); }


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteVertexArray", CallingConvention = CallingConvention.Cdecl)]
    private static extern void deleteVertexArray(uint array);
    protected override void DeleteVertexArray(uint array) { deleteVertexArray(array); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "deleteBuffer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void deleteBuffer(uint buffer);
    protected override void DeleteBuffer(uint buffer) { deleteBuffer(buffer); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getAttributeLocation", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint getAttributeLocation(uint shader, string attributeParameter);
    protected override uint GetAttributeLocation(uint shader, string attributeParameter) { return getAttributeLocation(shader, attributeParameter); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getUniformLocation", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint getUniformLocation(uint shader, string name);
    protected override uint GetUniformLocation(uint shader, string name) { return getUniformLocation(shader, name); }


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform1i", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint uniform1(uint location, int value);
    protected override void Uniform1(uint location, int value) { uniform1(location, value); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform1f", CallingConvention = CallingConvention.Cdecl)]
    public static extern void uniform1(uint location, float value);
    protected override void Uniform1(uint location, float value) { uniform1(location, value); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform2f", CallingConvention = CallingConvention.Cdecl)]
    public static extern void uniform2(uint location, float v1, float v2);
    protected override void Uniform2(uint location, float v1, float v2) { uniform2(location, v1, v2); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform3f", CallingConvention = CallingConvention.Cdecl)]
    public static extern void uniform3(uint location, float v1, float v2, float v3);
    protected override void Uniform3(uint location, float v1, float v2, float v3) { uniform3(location, v1, v2, v3); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform4f", CallingConvention = CallingConvention.Cdecl)]
    public static extern void uniform4(uint location, float v1, float v2, float v3, float v4);
    protected override void Uniform4(uint location, float v1, float v2, float v3, float v4) { uniform4(location, v1, v2, v3, v4); }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniformMatrix4f", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe void uniformMatrix4(uint location, bool transpose, float* values);

    protected override unsafe void UniformMatrix4(uint location, bool transpose, Matrix4 value)
    {
        fixed (float* values = value.Elements)
        {
            uniformMatrix4(location, transpose, values);
        }
    }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform1fv", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe void uniform1fv(uint location, int count, float* values);
    protected override unsafe void Uniform1(uint location, float[] values)
    {
        fixed (float* v = values)
        {
            uniform1fv(location, values.Length, v);
        }
    }

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "uniform3fv", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe void uniform3fv(uint location, int count, float* values);
    protected override unsafe void Uniform3(uint location, float[] values)
    {
        fixed (float* v = values)
        {
            uniform3fv(location, values.Length, v);
        }
    }


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getTexImage", CallingConvention = CallingConvention.Cdecl)]
    public static extern unsafe void getTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, void* pixels);
    protected override unsafe void GetTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, IntPtr pixels) { getTexImage(target, level, format, type, pixels.ToPointer()); }
}