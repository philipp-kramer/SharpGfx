using System;
using SharpGfx.OpenGL.Shading;

namespace SharpGfx.OpenGL
{
    internal class OglRenderObject : RenderObject
    {
        private readonly int _vertexCount;


        internal readonly uint Handle;
        private readonly VertexBuffer[] _buffers;

        public OglRenderObject(Space space, string name, OpenGlMaterial material, params VertexAttribute[] attributes)
            : base(space, name, material)
        {
            _buffers = new VertexBuffer[attributes.Length];
            _vertexCount = attributes[0].Values.Length / attributes[0].Stride;

            for (int i = 0; i < attributes.Length; i++)
            {
                var attribute = attributes[i];

                if (_vertexCount != attribute.Values.Length / attribute.Stride) throw new InvalidOperationException("all attributes must be for the same number of vertices");

                _buffers[i] = new OglVertexBuffer<float>((float[]) attribute.Values); // TODO: support other types
            }

            Handle = GL.GenVertexArray();
            material.SetVertexArrayAttributes(Handle, attributes, _buffers);
        }

        public override void Render()
        {
            GL.BindVertexArray(Handle);
            Draw();
            GL.BindVertexArray(0);
        }

        protected virtual void Draw()
        {
            GL.DrawTriangles(0, _vertexCount);
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteVertexArray(Handle);
        }

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();

            if (disposing)
            {
                foreach (var buffer in _buffers)
                {
                    buffer.Dispose();
                }
            }
        }

        ~OglRenderObject()
        {
            Dispose(false);
        }
    }
}
