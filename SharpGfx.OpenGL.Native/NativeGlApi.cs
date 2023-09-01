using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpGfx.Primitives;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx.OpenGL.Native;

public partial class NativeGlApi : GlApi
{
    internal const string OpenGlLibarary = @"x64/OpenGL.dll";

    internal NativeGlApi() {}

    [LibraryImport(OpenGlLibarary)]
    private static partial void viewport(int x, int y, int width, int height);
    protected override void Viewport(int x, int y, int width, int height) { viewport(x, y, width, height); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void clearColor(float r, float g, float b, float a);
    protected override void ClearColor(float r, float g, float b, float a) { clearColor(r, g, b, a); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void clear(GlBufferBit bufferBit);
    protected override void Clear(GlBufferBit bufferBit) { clear(bufferBit); }


    [LibraryImport(OpenGlLibarary)]
    private static partial uint genVertexArray();
    protected override uint GenVertexArray() { return genVertexArray(); }

    [LibraryImport(OpenGlLibarary)]
    private static partial uint genBuffer();
    protected override uint GenBuffer() { return genBuffer(); }

    [LibraryImport(OpenGlLibarary)]
    private static unsafe partial void bufferData(GlBufferTarget target, long size, void* data);
    protected override unsafe void BufferData(GlBufferTarget target, int size, nint data) { bufferData(target, size, data.ToPointer()); }

    [LibraryImport(OpenGlLibarary)]
    private static partial long getBufferParameter(GlBufferTarget target, GlBufferParameterName name);
    protected override long GetBufferParameter(GlBufferTarget target, GlBufferParameterName name) { return getBufferParameter(target, name); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void floatVertexAttribPointer(uint index, int size, int stride, int offset);
    protected override void FloatVertexAttribPointer(uint index, int size, int stride, int offset) { floatVertexAttribPointer(index, size, stride * sizeof(float), offset * sizeof(float)); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void bindVertexArray(uint array);
    protected override void BindVertexArray(uint array) { bindVertexArray(array); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void bindBuffer(GlBufferTarget target, uint buffer);
    protected override void BindBuffer(GlBufferTarget target, uint buffer) { bindBuffer(target, buffer); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void enableVertexAttribArray(uint array);
    protected override void EnableVertexAttribArray(uint array) { enableVertexAttribArray(array); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void drawTriangles(int first, int count);
    protected override void DrawTriangles(int count) { drawTriangles(0, count); }

    [LibraryImport(OpenGlLibarary)]
    private static unsafe partial void drawIndexedTriangles(int count, GlType type, void* indices);
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
    protected override unsafe void DrawIndexedTriangles<T>(int count, nint indices) { drawIndexedTriangles(count, GetElementsType<T>(), indices.ToPointer()); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void enable(GlCap cap);
    protected override void Enable(GlCap cap) { enable(cap); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void disable(GlCap cap);
    protected override void Disable(GlCap cap) { disable(cap); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void blendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor);
    protected override void BlendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor) { blendFunc(srcFactor, dstFactor); }


    [LibraryImport(OpenGlLibarary)]
    private static partial uint genTexture();
    protected override uint GenTexture() { return genTexture(); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void bindTexture(GlTextureTarget target, uint texture);
    protected override void BindTexture(GlTextureTarget target, uint texture) { bindTexture(target, texture); }

    [LibraryImport(OpenGlLibarary)]
    private static unsafe partial void texImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, void* pixels);

    protected override unsafe void TexImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, nint pixels) { texImage2D(target, level, width, height, border, format, type, pixels.ToPointer()); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void generateMipmap(GlTextureTarget target);
    protected override void GenerateMipmap(GlTextureTarget target) { generateMipmap(target); }
        
    [LibraryImport(OpenGlLibarary)]
    private static partial void textureParameterI(GlTextureTarget target, GlTextureParameterName name, int parameter);
    protected override void TextureParameterI(GlTextureTarget target, GlTextureParameterName name, GlTextureParameter parameter) { textureParameterI(target, name, (int) parameter); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void activeTexture(GlTextureUnit glTexture);
    protected override void ActiveTexture(GlTextureUnit unit) { activeTexture(unit); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void deleteTexture(uint texture);
    protected override void DeleteTexture(uint texture) { deleteTexture(texture); }


    [LibraryImport(OpenGlLibarary)]
    private static partial uint genFramebuffer();
    protected override uint GenFramebuffer() { return genFramebuffer(); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void bindFramebuffer(GlFramebufferTarget target, uint framebuffer);
    protected override void BindFramebuffer(GlFramebufferTarget target, uint framebuffer) { bindFramebuffer(target, framebuffer); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void deleteFramebuffer(uint framebuffer);
    protected override void DeleteFramebuffer(uint framebuffer) { deleteFramebuffer(framebuffer); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void framebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget textarget, uint texture, int level);
    protected override void FramebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget texTarget, uint texture, int level) { framebufferTexture2D(target, attachment, texTarget, texture, level); }

    [LibraryImport(OpenGlLibarary)]
    private static partial uint genRenderbuffer();
    protected override uint GenRenderbuffer() { return genRenderbuffer(); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void bindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer);
    protected override void BindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer) { bindRenderbuffer(target, renderbuffer); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void renderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalformat, int width, int height);
    protected override void RenderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalFormat, int width, int height) { renderbufferStorage(target, internalFormat, width, height); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void framebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbuffertarget, uint renderbuffer);
    protected override void FramebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbufferTarget, uint renderbuffer) { framebufferRenderbuffer(target, attachment, renderbufferTarget, renderbuffer); }
        
    [LibraryImport(OpenGlLibarary)]
    private static partial GlFramebufferErrorCode checkFramebufferStatus(GlFramebufferTarget target);
    protected override GlFramebufferErrorCode CheckFramebufferStatus(GlFramebufferTarget target) { return checkFramebufferStatus(target); }
        
    [LibraryImport(OpenGlLibarary)]
    private static partial void deleteRenderbuffer(uint renderbuffer);
    protected override void DeleteRenderbuffer(uint renderbuffer) { deleteRenderbuffer(renderbuffer); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void drawBuffer(uint buf);
    protected override void DrawBuffer(GlDrawBufferMode buffer) { drawBuffer((uint) buffer); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void readBuffer(uint src);
    protected override void ReadBuffer(GlReadBufferMode src) { readBuffer((uint) src); }

        
    [LibraryImport(OpenGlLibarary, StringMarshalling = StringMarshalling.Utf8)]
    private static partial uint compile(string vertexShaderSource, string fragmentShaderSource);
    protected override uint Compile(string vertexShaderSource, string fragmentShaderSource, string fragColorChannel, List<string> errors) { return compile(vertexShaderSource, fragmentShaderSource); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void useProgram(uint shaderProgram);
    protected override void UseProgram(uint shaderProgram) { useProgram(shaderProgram); }

    [LibraryImport(OpenGlLibarary)]
    public static partial void deleteProgram(uint handle);
    protected override void DeleteProgram(uint handle) { deleteProgram(handle); }


    [LibraryImport(OpenGlLibarary)]
    private static partial void deleteVertexArray(uint array);
    protected override void DeleteVertexArray(uint array) { deleteVertexArray(array); }

    [LibraryImport(OpenGlLibarary)]
    private static partial void deleteBuffer(uint buffer);
    protected override void DeleteBuffer(uint buffer) { deleteBuffer(buffer); }

    [LibraryImport(OpenGlLibarary, StringMarshalling = StringMarshalling.Utf8)]
    public static partial uint getAttributeLocation(uint shader, string attributeParameter);
    protected override uint GetAttributeLocation(uint shader, string attributeParameter) { return getAttributeLocation(shader, attributeParameter); }

    [LibraryImport(OpenGlLibarary, StringMarshalling = StringMarshalling.Utf8)]
    public static partial uint getUniformLocation(uint shader, string name);
    protected override uint GetUniformLocation(uint shader, string name) { return getUniformLocation(shader, name); }


    [LibraryImport(OpenGlLibarary)]
    public static partial uint uniform1i(uint location, int value);
    protected override void Uniform1(uint location, int value) { uniform1i(location, value); }

    [LibraryImport(OpenGlLibarary)]
    public static partial void uniform1f(uint location, float value);
    protected override void Uniform1(uint location, float value) { uniform1f(location, value); }

    [LibraryImport(OpenGlLibarary)]
    public static partial void uniform2f(uint location, float v1, float v2);
    protected override void Uniform2(uint location, float v1, float v2) { uniform2f(location, v1, v2); }

    [LibraryImport(OpenGlLibarary)]
    public static partial void uniform3f(uint location, float v1, float v2, float v3);
    protected override void Uniform3(uint location, float v1, float v2, float v3) { uniform3f(location, v1, v2, v3); }

    [LibraryImport(OpenGlLibarary)]
    public static partial void uniform4f(uint location, float v1, float v2, float v3, float v4);
    protected override void Uniform4(uint location, float v1, float v2, float v3, float v4) { uniform4f(location, v1, v2, v3, v4); }

    [LibraryImport(OpenGlLibarary)]
    public static unsafe partial void uniformMatrix4f(uint location, [MarshalAs(UnmanagedType.Bool)] bool transpose, float* values);

    protected override unsafe void UniformMatrix4(uint location, bool transpose, Matrix4 value)
    {
        fixed (float* values = value.Elements)
        {
            uniformMatrix4f(location, transpose, values);
        }
    }

    [LibraryImport(OpenGlLibarary)]
    public static unsafe partial void uniform1fv(uint location, int count, float* values);
    protected override unsafe void Uniform1(uint location, float[] values)
    {
        fixed (float* v = values)
        {
            uniform1fv(location, values.Length, v);
        }
    }

    [LibraryImport(OpenGlLibarary)]
    public static unsafe partial void uniform3fv(uint location, int count, float* values);
    protected override unsafe void Uniform3(uint location, float[] values)
    {
        fixed (float* v = values)
        {
            uniform3fv(location, values.Length, v);
        }
    }


    [LibraryImport(OpenGlLibarary)]
    public static unsafe partial void getTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, void* pixels);
    protected override unsafe void GetTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, nint pixels) { getTexImage(target, level, format, type, pixels.ToPointer()); }
}