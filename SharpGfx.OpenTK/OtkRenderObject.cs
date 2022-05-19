using System;
using System.Linq;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal class OtkRenderObject : RenderObject, IDisposable
    {
        private readonly int _vertexCount;
        internal readonly int Handle;
        private readonly VertexAttribute[] _attributes;

        public OtkRenderObject(Space space, string name, Material material, params VertexAttribute[] attributes)
            : base(space, name, material)
        {
            _attributes = attributes;
            _vertexCount = GetVertexCount(attributes[0]);
            for (int i = 1; i < attributes.Length; i++)
            {
                if (_vertexCount != GetVertexCount(attributes[i]))
                {
                    throw new InvalidOperationException("all attributes must be for the same number of vertices");
                }
            }

            Handle = GL.GenVertexArray();
            var shading = ((OtkShadedMaterial)material).Shading;

            shading.DoInContext(() =>
            {
                GL.BindVertexArray(Handle);

                foreach (var attribute in _attributes)
                {
                    GL.BindBuffer(BufferTarget.ArrayBuffer, ((OtkVertexBuffer<float>)attribute.Buffer).Handle);

                    var location = shading.GetAttributeHandle(attribute.Parameter);
                    GL.EnableVertexAttribArray(location);
                    GL.VertexAttribPointer(location, attribute.Rank, VertexAttribPointerType.Float, false, attribute.Rank * sizeof(float), 0);

                    GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
                }

                GL.BindVertexArray(0);
            });
        }

        private static int GetVertexCount(VertexAttribute attribute)
        {
            return (int) (attribute.Buffer.Length / attribute.Rank);
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

        public override void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();

            if (disposing)
            {
                foreach (var buffer in _attributes.Select(a => a.Buffer).Cast<IDisposable>())
                {
                    buffer.Dispose();
                }
            }
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteVertexArray(Handle);
        }

        ~OtkRenderObject()
        {
            UnmanagedRelease.Add(() => Dispose(false));
        }
    }
}
