using System;
using System.Collections.Generic;
using SharpGfx.Primitives;

namespace SharpGfx;

public abstract class Rendering : IDisposable
{
    protected internal Device Device { get; }
    public Color3 Background { get; }
    public List<Instance> Scene { get; }

    protected Rendering(Device device, Color3 background)
    {
        Device = device;
        Background = background;
        Scene = new List<Instance>();
    }

    public void OnLoad()
    {
        Device.CheckSpaces(Scene);
    }

    public virtual void OnMouseUp(MouseButton mouseButton) { }

    public virtual void OnUpdateFrame() { }

    public virtual void OnRenderFrame(Window window)
    {
        Device.Render(window, this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var instance in Scene)
            {
                if (instance is IDisposable disposable)
                {
                    disposable.Dispose();
                }
            }
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