using SharpGfx.OpenGL.Materials;

namespace SharpGfx.OpenGL;

internal sealed class GlIndexedSurface<T> : GlSurface
    where T : struct
{
    private readonly GlApi _gl;
    private readonly GlIndexBuffer<T> _indexBuffer;

    public GlIndexedSurface(GlApi gl, OpenGlMaterial material, T[] triangles, params SurfaceAttribute[] attributes) 
        : base(gl, material, attributes)
    {
        _gl = gl;
        _indexBuffer = new GlIndexBuffer<T>(gl, triangles);
    }

    internal override void Draw()
    {
        _gl.BindVertexArray(VertexArray);
        _gl.BindBuffer(GlBufferTarget.ElementArrayBuffer, _indexBuffer.Handle);
        _gl.DrawIndexedTriangles<T>(_indexBuffer.Length, nint.Zero);
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

    ~GlIndexedSurface()
    {
        _gl.Add(() => Dispose(false));
    }
}