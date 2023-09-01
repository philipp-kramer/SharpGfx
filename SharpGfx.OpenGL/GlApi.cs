using System;
using System.Collections.Generic;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL;

public abstract class GlApi
{
    private readonly Queue<Action> _queue = new Queue<Action>();
        
    protected internal void Add(Action pending)
    {
        _queue.Enqueue(pending);
    }

    protected internal void ExecutePending()
    {
        while (_queue.Count > 0)
        {
            _queue.Dequeue()?.Invoke();
        }
    }

    protected internal abstract void Viewport(int x, int y, int width, int height);
    protected internal abstract void ClearColor(float r, float g, float b, float a);
    protected internal abstract void Clear(GlBufferBit bufferBit);

    protected internal abstract uint GenVertexArray();
    protected internal abstract uint GenBuffer();
    protected internal abstract void BufferData(GlBufferTarget target, int size, nint data);
    protected internal abstract long GetBufferParameter(GlBufferTarget target, GlBufferParameterName name);
    protected internal abstract void FloatVertexAttribPointer(uint index, int size, int stride, int offset);
    protected internal abstract void BindVertexArray(uint array);
    protected internal abstract void BindBuffer(GlBufferTarget target, uint buffer);
    protected internal abstract void EnableVertexAttribArray(uint array);
    protected internal abstract void DrawTriangles(int count);
    protected internal abstract void DrawIndexedTriangles<T>(int count, nint indices);
    protected internal abstract void Enable(GlCap cap);
    protected internal abstract void Disable(GlCap cap);
    protected internal abstract void BlendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor);

    protected internal abstract uint GenTexture();
    protected internal abstract void BindTexture(GlTextureTarget target, uint texture);
    protected internal abstract void TexImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, nint pixels);
    protected internal abstract void GenerateMipmap(GlTextureTarget target);

    protected internal abstract void TextureParameterI(GlTextureTarget target, GlTextureParameterName name, GlTextureParameter parameter);
    protected internal abstract void ActiveTexture(GlTextureUnit unit);
    protected internal abstract void DeleteTexture(uint texture);
    public void ClearTexture(int unit)
    {
        BindTexture(GlTextureTarget.Texture2D, 0);
    }

    protected internal abstract void GetTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, nint pixels);

    protected internal abstract uint GenFramebuffer();
    protected internal abstract void BindFramebuffer(GlFramebufferTarget target, uint framebuffer);
    protected internal abstract void DeleteFramebuffer(uint framebuffer);
    protected internal abstract void FramebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget texTarget, uint texture, int level);
    protected internal abstract uint GenRenderbuffer();
    protected internal abstract void BindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer);
    protected internal abstract void RenderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalFormat, int width, int height);
    protected internal abstract void FramebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbufferTarget, uint renderbuffer);
    protected internal abstract GlFramebufferErrorCode CheckFramebufferStatus(GlFramebufferTarget target);
    protected internal abstract void DeleteRenderbuffer(uint renderbuffer);
    protected internal abstract void DrawBuffer(GlDrawBufferMode buffer);
    protected internal abstract void ReadBuffer(GlReadBufferMode src);

    protected internal abstract uint Compile(string vertexShaderSource, string fragmentShaderSource, string fragColorChannel, List<string> errors);
    protected internal abstract void UseProgram(uint shaderProgram);
    protected internal abstract void DeleteProgram(uint handle);

    protected internal abstract void DeleteVertexArray(uint array);
    protected internal abstract void DeleteBuffer(uint buffer);
    protected internal abstract uint GetAttributeLocation(uint shader, string attributeParameter);
    protected internal abstract uint GetUniformLocation(uint shader, string name);

    protected internal abstract void Uniform1(uint location, int value);
    protected internal abstract void Uniform1(uint location, float value);
    protected internal abstract void Uniform2(uint location, float v1, float v2);
    protected internal abstract void Uniform3(uint location, float v1, float v2, float v3);
    protected internal abstract void Uniform4(uint location, float v1, float v2, float v3, float v4);
    protected internal abstract void UniformMatrix4(uint location, bool transpose, Matrix4 values);
    protected internal abstract void Uniform1(uint location, float[] values);
    protected internal abstract void Uniform3(uint location, float[] values);
}