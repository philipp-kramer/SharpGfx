using System;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal sealed class OtkIndexedRenderObject<T> : OtkRenderObject
        where T : struct
    {
        private readonly OtkIndexBuffer<T> _indexBuffer;

        public OtkIndexedRenderObject(Space space, Material material, T[] indices, params VertexAttribute[] attributes) 
            : base(space, material, attributes)
        {
            _indexBuffer = new OtkIndexBuffer<T>(indices);
        }

        protected override void Draw()
        {
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, _indexBuffer.Handle);
            GL.DrawElements(PrimitiveType.Triangles, _indexBuffer.Length, GetElementsType(), 0);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, 0);
        }

        private DrawElementsType GetElementsType()
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