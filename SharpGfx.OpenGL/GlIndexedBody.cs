using System;
using SharpGfx.OpenGL.Materials;

namespace SharpGfx.OpenGL;

internal sealed class GlIndexedBody<T> : GlBody
    where T : struct
{
    private readonly GlApi _gl;
    private readonly GlIndexBuffer<T> _indexBuffer;

    public GlIndexedBody(GlApi gl, OpenGlMaterial material, T[] triangles, params IVertexAttribute[] attributes) 
        : base(gl, material, attributes)
    {
        _gl = gl;
        _indexBuffer = new GlIndexBuffer<T>(gl, triangles);
    }

    internal override void Draw()
    {
        _gl.BindVertexArray(VertexArray);
        _gl.BindBuffer(GlBufferTarget.ElementArrayBuffer, _indexBuffer.Handle);
        _gl.DrawIndexedTriangles<T>(_indexBuffer.Length, IntPtr.Zero);
        _gl.BindBuffer(GlBufferTarget.ElementArrayBuffer, 0);
        _gl.BindVertexArray(0);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _indexBuffer.Dispose();
        }
        base.Dispose(disposing);
    }

    ~GlIndexedBody()
    {
        _gl.Add(() => Dispose(false));
    }
}