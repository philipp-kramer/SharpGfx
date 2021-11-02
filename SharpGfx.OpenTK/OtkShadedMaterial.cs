using OpenTK.Graphics.OpenGL;
using SharpGfx.Primitives;
using System;

namespace SharpGfx.OpenTK
{
    public abstract class OtkShadedMaterial : Material, IDisposable
    {
        public bool Transparent { get; set; }
        protected internal OtkShading Shading { get; }

        protected OtkShadedMaterial(string vertexShader, string fragmentShader, bool hasTexture)
        {
            Shading = new OtkShading(vertexShader, fragmentShader, hasTexture);
        }

        protected internal virtual void Apply(Point3 cameraPosition)
        {
            if (Transparent)
            {
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            }
        }

        protected internal virtual void UnApply()
        {
            if (Transparent)
            {
                GL.Disable(EnableCap.Blend);
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                Shading?.Dispose();
            }
        }

        ~OtkShadedMaterial()
        {
            UnmanagedRelease.Add(() => Dispose(false));
        }
    }
}