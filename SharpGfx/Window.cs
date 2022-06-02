﻿using SharpGfx.Host;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class Window
    {
        public static readonly Space Screen = new HostSpace(Domain.View);

        protected Rendering Rendering { get; private set; }

        public virtual void Show(Rendering rendering)
        {
            Rendering = rendering;
        }

        protected virtual void OnLoad()
        {
            Rendering.OnLoad();
        }

        protected void OnResize(IVector2 size)
        {
            Rendering.OnResize(size);
        }

        protected void OnUpdateFrame()
        {
            Rendering.OnUpdateFrame();
        }

        protected void OnRenderFrame()
        {
            Rendering.OnRenderFrame();
        }
    }
}
