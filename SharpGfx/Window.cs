using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class Window
    {
        private CameraRendering _rendering;

        public event Action<MouseButtons> MouseUp;

        public virtual void Show(CameraRendering rendering)
        {
            _rendering = rendering;
        }

        protected void OnKeyDown(ConsoleKey key)
        {
            _rendering.Camera.OnKeyDown(key);
        }

        protected void MouseMoving(Vector2 delta, MouseButtons mouseButton)
        {
            _rendering.Camera.MouseMoving(delta, mouseButton);
        }

        protected void InvokeMouseUp(MouseButtons buttonClicked)
        {
            MouseUp?.Invoke(buttonClicked);
        }

        protected virtual void OnLoad()
        {
            _rendering?.OnLoad();
        }

        protected void OnResize(Vector2 size)
        {
            _rendering?.OnResize(size);
        }

        protected void OnUpdateFrame()
        {
            _rendering.OnUpdateFrame();
        }

        protected void OnRenderFrame()
        {
            _rendering.OnRenderFrame();
        }
    }
}
