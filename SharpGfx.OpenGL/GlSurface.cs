using System;
using SharpGfx.OpenGL.Materials;

namespace SharpGfx.OpenGL;

internal class GlSurface : Surface
{
    private readonly GlApi _gl;
    internal uint[] VertexBuffers { get; }
    internal uint VertexArray { get; }

    public GlSurface(GlApi gl, OpenGlMaterial material, params SurfaceAttribute[] attributes)
        : base(material, attributes[0].Values.Length / attributes[0].Rank)
    {
        _gl = gl;
        VertexBuffers = new uint[attributes.Length];

        for (int i = 0; i < attributes.Length; i++)
        {
            var attribute = attributes[i];
            if (attribute.Values.Length / attribute.Rank != VertexCount) throw new ArgumentException(nameof(attributes));

            VertexBuffers[i] = GlBuffer.CreateBuffer(gl, attribute.Values, GlBufferTarget.ArrayBuffer);
        }

        VertexArray = gl.GenVertexArray();
        material.SetVertexArrayAttributes(VertexArray, attributes, VertexBuffers);
    }

    internal virtual void Draw()
    {
        _gl.BindVertexArray(VertexArray);
        _gl.DrawTriangles(VertexCount);
        _gl.BindVertexArray(0);
    }

    private void ReleaseUnmanagedResources()
    {
        _gl.DeleteVertexArray(VertexArray);
        foreach (var t in VertexBuffers)
        {
            _gl.DeleteBuffer(t);
        }
    }

    protected override void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
    }

    ~GlSurface()
    {
        _gl.Add(() => Dispose(false));
    }
}