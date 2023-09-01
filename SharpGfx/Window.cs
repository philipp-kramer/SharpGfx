namespace SharpGfx;

public abstract class Window
{
    public int Width { get; protected set; }
    public int Height { get; protected set; }

    protected Rendering? Rendering { get; private set; }

    protected Window(int width, int height)
    {
        Width = width;
        Height = height;
    }

    public virtual void Show(Rendering rendering)
    {
        Rendering = rendering;
    }

    protected virtual void OnLoad()
    {
        Rendering!.OnLoad();
    }

    protected void OnUpdateFrame()
    {
        Rendering!.OnUpdateFrame();
    }

    protected void OnRenderFrame()
    {
        Rendering!.OnRenderFrame(this);
    }
}