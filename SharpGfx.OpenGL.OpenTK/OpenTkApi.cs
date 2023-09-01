using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenGL.OpenTK;

public class OpenTkApi : GlApi
{
    internal OpenTkApi() {}

    protected override void Viewport(int x, int y, int width, int height) { GL.Viewport(x, y, width, height); }

    protected override void ClearColor(float r, float g, float b, float a) { GL.ClearColor(r, g, b, a); }

    protected override void Clear(GlBufferBit bufferBit) { GL.Clear((ClearBufferMask) bufferBit); }

    protected override uint GenVertexArray() { return (uint) GL.GenVertexArray(); }

    protected override uint GenBuffer() { return (uint) GL.GenBuffer(); }

    protected override void BufferData(GlBufferTarget target, int size, nint data) { GL.BufferData((BufferTarget) target, size, data, BufferUsageHint.StaticDraw); }

    protected override long GetBufferParameter(GlBufferTarget target, GlBufferParameterName name)
    {
        GL.GetBufferParameter((BufferTarget) target, (BufferParameterName) name, out long value);
        return value;
    }

    protected override void FloatVertexAttribPointer(uint index, int size, int stride, int offset)
    {
        GL.VertexAttribPointer((int) index, size, VertexAttribPointerType.Float, false, stride * sizeof(float), offset * sizeof(float));
    }

    protected override void BindVertexArray(uint array) { GL.BindVertexArray(array); }

    protected override void BindBuffer(GlBufferTarget target, uint buffer) { GL.BindBuffer((BufferTarget) target, buffer); }

    protected override void EnableVertexAttribArray(uint array) { GL.EnableVertexAttribArray(array); }

    protected override void DrawTriangles(int count) { GL.DrawArrays(PrimitiveType.Triangles, 0, count); }

    private DrawElementsType GetElementsType<T>()
    {
        if (typeof(T) == typeof(byte))
        {
            return DrawElementsType.UnsignedByte;
        }
        else if (typeof(T) == typeof(ushort))
        {
            return DrawElementsType.UnsignedShort;
        }
        else if (typeof(T) == typeof(uint))
        {
            return DrawElementsType.UnsignedInt;
        }
        else
        {
            throw new NotSupportedException($"type {typeof(T)} not supported for indexing");
        }
    }

    protected override void DrawIndexedTriangles<T>(int count, nint indices) { GL.DrawElements(PrimitiveType.Triangles, count, GetElementsType<T>(), indices); }

    protected override void Enable(GlCap cap) { GL.Enable((EnableCap) cap);
    }

    protected override void Disable(GlCap cap) { GL.Disable((EnableCap) cap); }

    protected override void BlendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor) { GL.BlendFunc((BlendingFactor) srcFactor, (BlendingFactor) dstFactor); }

    protected override uint GenTexture() { return (uint) GL.GenTexture(); }

    protected override void BindTexture(GlTextureTarget target, uint texture) { GL.BindTexture((TextureTarget) target, texture); }

    protected override void TexImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, nint pixels)
    {
        GL.TexImage2D((TextureTarget) target, level, GetInternal(format), width, height, border, (PixelFormat) format, (PixelType) type, pixels);
    }

    protected override void GenerateMipmap(GlTextureTarget target) { GL.GenerateMipmap((GenerateMipmapTarget) target); }

    protected override void TextureParameterI(GlTextureTarget target, GlTextureParameterName name, GlTextureParameter parameter) { GL.TexParameter((TextureTarget) target, (TextureParameterName) name, (int) parameter); }

    protected override void ActiveTexture(GlTextureUnit unit) { GL.ActiveTexture((TextureUnit) unit); }

    protected override void DeleteTexture(uint texture) { GL.DeleteTexture(texture); }

    protected override void GetTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, nint pixels) { GL.GetTexImage((TextureTarget)target, level, (PixelFormat)format, (PixelType)type, pixels); }

    protected override uint GenFramebuffer() { return (uint) GL.GenFramebuffer(); }

    protected override void BindFramebuffer(GlFramebufferTarget target, uint framebuffer) {  GL.BindFramebuffer((FramebufferTarget) target, framebuffer); }

    protected override void DeleteFramebuffer(uint framebuffer) { GL.DeleteFramebuffer(framebuffer); }

    protected override void FramebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget texTarget, uint texture, int level) { GL.FramebufferTexture2D((FramebufferTarget) target, (FramebufferAttachment) attachment, (TextureTarget) texTarget, texture, level); }

    protected override uint GenRenderbuffer() {  return (uint)GL.GenRenderbuffer(); }

    protected override void BindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer) {  GL.BindRenderbuffer((RenderbufferTarget) target, renderbuffer); }

    protected override void RenderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalFormat, int width, int height)  { GL.RenderbufferStorage((RenderbufferTarget) target, (RenderbufferStorage) internalFormat, width, height); }

    protected override void FramebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbufferTarget, uint renderbuffer) { GL.FramebufferRenderbuffer((FramebufferTarget) target, (FramebufferAttachment) attachment, (RenderbufferTarget) renderbufferTarget, renderbuffer); }

    protected override GlFramebufferErrorCode CheckFramebufferStatus(GlFramebufferTarget target) { return (GlFramebufferErrorCode) GL.CheckFramebufferStatus((FramebufferTarget) target); }

    protected override void DeleteRenderbuffer(uint renderbuffer) {  GL.DeleteRenderbuffer(renderbuffer); }

    protected override void DrawBuffer(GlDrawBufferMode buffer) {  GL.DrawBuffer((DrawBufferMode) buffer); }

    protected override void ReadBuffer(GlReadBufferMode src) {  GL.ReadBuffer((ReadBufferMode) src); }

    protected override uint Compile(string vertexShaderSource, string fragmentShaderSource, string fragColorChannel, List<string> errors)  { return Compilation.Compile(vertexShaderSource, fragmentShaderSource, fragColorChannel, errors); }

    protected override void UseProgram(uint shaderProgram) { GL.UseProgram(shaderProgram); }

    protected override void DeleteProgram(uint handle)
    {
        Add(() => GL.DeleteProgram(handle)); // can be called from finalizers
    }

    protected override void DeleteVertexArray(uint array) { GL.DeleteVertexArray(array); }

    protected override void DeleteBuffer(uint buffer) { GL.DeleteBuffer(buffer); }

    protected override uint GetAttributeLocation(uint shader, string attributeParameter) { return (uint) GL.GetAttribLocation(shader, attributeParameter); }

    protected override uint GetUniformLocation(uint shader, string name) { return (uint) GL.GetUniformLocation(shader, name); }

    protected override void Uniform1(uint location, int value) { GL.Uniform1((int) location, value); }

    protected override void Uniform1(uint location, float value) { GL.Uniform1((int) location, value); }

    protected override void Uniform2(uint location, float v1, float v2) { GL.Uniform2((int) location, v1, v2); }

    protected override void Uniform3(uint location, float v1, float v2, float v3) { GL.Uniform3((int) location, v1, v2, v3); }

    protected override void Uniform4(uint location, float v1, float v2, float v3, float v4) { GL.Uniform4((int) location, v1, v2, v3, v4); }

    protected override void Uniform1(uint location, float[] values) { GL.Uniform1((int) location, values.Length, values); }
    protected override void Uniform3(uint location, float[] values) { GL.Uniform3((int) location, values.Length, values); }

    protected override void UniformMatrix4(uint location, bool transpose, Primitives.Matrix4 values) 
    {
        var v = ((Matrix4) values).Value;
        GL.UniformMatrix4((int) location, transpose, ref v);
    }

    private PixelInternalFormat GetInternal(GlPixelFormat format)
    {
        return format switch
        {
            GlPixelFormat.Rgb => PixelInternalFormat.Rgb,
            GlPixelFormat.Rgba => PixelInternalFormat.Rgba,
            GlPixelFormat.DepthComponent => PixelInternalFormat.DepthComponent,
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, default)
        };
    }
}