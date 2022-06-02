using System;
using OpenTK.Graphics.OpenGL;
using SharpGfx.OpenGL.Shading;

namespace SharpGfx.OpenTK
{
    internal class OtkRenderObject : RenderObject
    {
        private readonly int _vertexCount;
        internal readonly uint Handle;
        private readonly VertexBuffer[] _buffers;

        public OtkRenderObject(Space space, string name, OpenGlMaterial material, params VertexAttribute[] attributes)
            : base(space, name, material)
        {
            _vertexCount = attributes[0].Values.Length / attributes[0].Stride;
            _buffers = new VertexBuffer[attributes.Length];

            for (int i = 0; i < attributes.Length; i++)
            {
                var attribute = attributes[i];

                if (_vertexCount != attribute.Values.Length / attribute.Stride) throw new InvalidOperationException("all attributes must be for the same number of vertices");

                _buffers[i] = new OtkVertexBuffer<float>((float[]) attribute.Values); // TODO: support other types
            }

            Handle = (uint) GL.GenVertexArray();
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
            GL.DrawArrays(PrimitiveType.Triangles, 0, _vertexCount);
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

        ~OtkRenderObject()
        {
            UnmanagedRelease.Add(() => Dispose(false));
        }
    }
}
