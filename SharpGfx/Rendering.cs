using System;
using System.Collections.Generic;
using System.Drawing;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public class Rendering : IDisposable
    {
        protected static float Fovy { get; } = MathF.PI / 4;

        protected Device Device { get; }
        protected Size Size { get; private set; }
        protected Color3 AmbientColor { get; }
        public List<RenderObject> Scene { get; }

        protected Rendering(Device device, Size size, Color3 ambientColor)
        {
            Device = device;
            Size = size;
            AmbientColor = ambientColor;
            Scene = new List<RenderObject>();
        }

        public void OnLoad()
        {
            Device.CheckSpaces(Scene);
        }

        public virtual void OnResize(Size size)
        {
            Size = size;
            Device.SetProjection(Scene, GetPerspectiveProjection());
        }

        protected Matrix4 GetPerspectiveProjection()
        {
            return Device.GetPerspectiveProjection(
                Fovy,
                (float)Size.Width / Size.Height,
                0.1f, 100.0f);
        }

        public virtual void OnUpdateFrame()
        {
        }

        public virtual void OnRenderFrame()
        {
            Device.Render(Scene, Size, default, AmbientColor.GetColor4(1));
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var @object in Scene)
                {
                    @object.Dispose();
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~Rendering()
        {
            Dispose(false);
        }
    }
}