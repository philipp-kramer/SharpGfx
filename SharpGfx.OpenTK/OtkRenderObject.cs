using System;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal class OtkRenderObject : RenderObject
    {
        private readonly int _vertexCount;
        internal readonly int Handle;
        private readonly VertexAttribute[] _attributes;

        public OtkRenderObject(Space space, Material material, params VertexAttribute[] attributes)
            : base(space, material)
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
            var shading = ((OtkShadedMaterial) Material).Shading;
            shading.Set("model", Transform);
            shading.CheckInputs();

            GL.BindVertexArray(Handle);
            Draw();
            GL.BindVertexArray(0);

            shading.ResetIdentityMatrix4("model");
        }

        protected virtual void Draw()
        {
            GL.DrawArrays(PrimitiveType.Triangles, 0, _vertexCount);
        }

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();

            if (disposing)
            {
                foreach (var attribute in _attributes)
                {
                    attribute.Buffer.Dispose();
                }
            }
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteVertexArray(Handle);
        }
    }
}
