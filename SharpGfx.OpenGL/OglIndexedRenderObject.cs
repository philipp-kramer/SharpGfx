using System;
using SharpGfx.OpenGL.Shading;

namespace SharpGfx.OpenGL
{
    internal sealed class OglIndexedRenderObject<T> : OglRenderObject
        where T : struct
    {
        private readonly OglIndexBuffer<T> _indexBuffer;

        public OglIndexedRenderObject(Space space, string name, OpenGlMaterial material, T[] triangles, params VertexAttribute[] attributes) 
            : base(space, name, material, attributes)
        {
            _indexBuffer = new OglIndexBuffer<T>(triangles);
        }

        protected override void Draw()
        {
            GL.BindBuffer(GlBufferTarget.ElementArrayBuffer, _indexBuffer.Handle);
            unsafe
            {
                GL.DrawIndexedTriangles(_indexBuffer.Length, GetElementsType(), (void*)0);
            }
            GL.BindBuffer(GlBufferTarget.ElementArrayBuffer, 0);
        }

        private GlType GetElementsType()
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

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _indexBuffer.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}