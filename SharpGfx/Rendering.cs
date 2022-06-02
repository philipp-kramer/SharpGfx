using System;
using System.Collections.Generic;
using System.Linq;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class Rendering : IDisposable
    {
        public const float Near = 0.1f;
        public const float Far = 100f;
        public const float FovY = MathF.PI / 4;

        protected Device Device { get; }
        protected IVector2 Size { get; private set; }
        protected Color3 AmbientColor { get; }
        public List<RenderObject> Scene { get; }

        protected Rendering(Device device, IVector2 size, Color3 ambientColor)
        {
            Device = device;
            Size = size;
            AmbientColor = ambientColor;
            Scene = new List<RenderObject>();
        }

        protected float Aspect => Size.X / Size.Y;

        public void OnLoad()
        {
            Device.CheckSpaces(Scene);
        }

        public virtual void OnMouseUp(MouseButtons mouseButton) { }

        public virtual void OnResize(IVector2 size)
        {
            Size = size;
        }

        public virtual void OnUpdateFrame() { }

        public virtual void OnRenderFrame()
        {
            Device.Render(Scene, Size, AmbientColor.GetColor4(1));
        }

        protected void DisposeObjects()
        {
            foreach (var @object in Scene)
            {
                if (@object is IDisposable disposable)
                {
                    disposable.Dispose();
                }
            }
        }

        protected void DisposeMaterials()
        {
            var materials = Scene
                .Select(o => o.Material)
                .Distinct();
            foreach (var material in materials)
            {
                if (material is IDisposable disposable)
                {
                    disposable.Dispose();
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                DisposeObjects();
                DisposeMaterials();
            }
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        ~Rendering()
        {
            Dispose(false);
        }
    }
}