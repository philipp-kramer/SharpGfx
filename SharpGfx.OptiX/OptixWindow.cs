using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using SharpGfx.OptiX.Materials;

namespace SharpGfx.OptiX;

public sealed class OptixWindow : Window, IDisposable
{
    private static readonly Dictionary<int, ConsoleKey> KeyMapping = new()
    {
        { 290, ConsoleKey.F1 },
        { 291, ConsoleKey.F2 },
        { 292, ConsoleKey.F3 },
        { 293, ConsoleKey.F4 },
        { 294, ConsoleKey.F5 },
        { 295, ConsoleKey.F6 },
        { 296, ConsoleKey.F7 },
        { 297, ConsoleKey.F8 },
        { 298, ConsoleKey.F9 },
        { 299, ConsoleKey.F10 },
        { 300, ConsoleKey.F11 },
        { 301, ConsoleKey.F12 },
        { 320, ConsoleKey.NumPad0 },
        { 321, ConsoleKey.NumPad1 },
        { 322, ConsoleKey.NumPad2 },
        { 323, ConsoleKey.NumPad3 },
        { 324, ConsoleKey.NumPad4 },
        { 325, ConsoleKey.NumPad5 },
        { 326, ConsoleKey.NumPad6 },
        { 327, ConsoleKey.NumPad7 },
        { 328, ConsoleKey.NumPad8 },
        { 329, ConsoleKey.NumPad9 }
    };

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private readonly struct Inputs
    {
        public readonly int Key;
        public readonly int Button;
        public readonly int Action;
        public readonly float MouseX;
        public readonly float MouseY;
        public readonly float ScrollX;
        public readonly float ScrollY;
    };

    [DllImport(@".\optix.dll", EntryPoint = "create_camera", CallingConvention = CallingConvention.StdCall)]
    private static extern CameraPtr CreateCamera();

    [DllImport(@".\optix.dll", EntryPoint = "build", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe int Build(ContextPtr context, GeometryPtr* geometries, MaterialPtr* materials, int bCount, InstancePtr* instances, int iCount, float* bgColor);

    [DllImport(@".\optix.dll", EntryPoint = "update", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe int Update(ContextPtr context, GeometryPtr* geometries, int bCount, InstancePtr* instances, int iCount);

    [DllImport(@".\optix.dll", EntryPoint = "get_inputs", CallingConvention = CallingConvention.StdCall)]
    private static extern Inputs GetInputs(WindowPtr window);

    [DllImport(@".\optix.dll", EntryPoint = "open_window", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe WindowPtr OpenWindow(ContextPtr context, byte* title, int width, int height);

    [DllImport(@".\optix.dll", EntryPoint = "window_should_close", CallingConvention = CallingConvention.StdCall)]
    private static extern bool WindowShouldClose(WindowPtr window);

    [DllImport(@".\optix.dll", EntryPoint = "close", CallingConvention = CallingConvention.StdCall)]
    private static extern int Close(WindowPtr window, CameraPtr camera, ContextPtr context);

    private readonly string _title;
    private readonly InteractiveCamera _camera;
    private readonly ButtonState _mouseLeftButton = new();
    private readonly ButtonState _mouseMiddleButton = new();
    private readonly ButtonState _mouseRightButton = new();
    private readonly ContextPtr _context;
    private OptixBody[] _bodies;

    internal WindowPtr Window { get; private set; }
    internal CameraPtr Camera { get; }

    public OptixWindow(string title, int width, int height, OptixDevice device, InteractiveCamera camera = null)
        : base(width, height)
    {
        _title = title;
        _camera = camera;
        Camera = CreateCamera();
        _context = device.Context;
    }

    public override unsafe void Show(Rendering rendering)
    {
        base.Show(rendering);

        var optixInstances = rendering
            .Scene
            .SelectMany(instance => instance.All)
            .OfType<OptixInstance>()
            .ToArray();

        var instanceHandles = optixInstances
            .Select(i => i.Create())
            .ToArray();

        _bodies = optixInstances
            .Select(instance => instance.Body)
            .Distinct()
            .ToArray();
        var geometryHandles = _bodies
            .Select(body => body.Handle)
            .ToArray();
        var materialHandles = _bodies
            .Select(body => body.Material)
            .Cast<OptixMaterial>()
            .Select(material => material.Handle)
            .ToArray();

        fixed (GeometryPtr* geometries = geometryHandles)
        fixed (MaterialPtr* materials = materialHandles)
        fixed (InstancePtr* instances = instanceHandles)
        fixed (float* bgColor = (float[]) rendering.Background.Vector.Values)
        {
            var error = Build(_context, geometries, materials, _bodies.Length, instances, optixInstances.Length, bgColor);
            if (error != 0) throw new Exception($"{error}");
        }

        rendering.OnLoad();

        fixed (byte* title = Encoding.ASCII.GetBytes(_title))
        {
            Window = OpenWindow(_context, title, Width, Height);
        }

        while (!WindowShouldClose(Window))
        {
            Update(geometryHandles, instanceHandles, _bodies.Length, optixInstances.Length);

            OnRenderFrame();
                
            HandleInputs();
        }
    }

    private void HandleInputs()
    {
        var inputs = GetInputs(Window);
        if (!KeyMapping.TryGetValue(inputs.Key, out var consoleKey))
        {
            consoleKey = (ConsoleKey)inputs.Key;
        }

        if (inputs.Key != 0) _camera?.OnKeyDown(consoleKey);

        if (inputs.Action == 1)
        {
            switch (inputs.Button)
            {
                case 0:
                    MouseAction(inputs, MouseButton.Left);
                    _mouseLeftButton.Down();
                    _mouseMiddleButton.Up();
                    _mouseRightButton.Up();
                    break;

                case 1:
                    MouseAction(inputs, MouseButton.Right);
                    _mouseRightButton.Down();
                    _mouseLeftButton.Up();
                    _mouseMiddleButton.Up();
                    break;

                case 2:
                    MouseAction(inputs, MouseButton.Middle);
                    _mouseMiddleButton.Down();
                    _mouseLeftButton.Up();
                    _mouseRightButton.Up();
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(inputs.Button));
            }
        }
        else
        {
            _mouseLeftButton.Up();
            _mouseMiddleButton.Up();
            _mouseRightButton.Up();
        }

        if (inputs.ScrollX != 0 || inputs.ScrollY != 0)
        {
            _camera?.MouseDragging(MouseButton.Middle, inputs.ScrollX, inputs.ScrollY);
        }

        MouseX = inputs.MouseX;
        MouseY = inputs.MouseY;
    }

    private unsafe void Update(GeometryPtr[] geometryHandles, InstancePtr[] instanceHandles, int bCount, int iCount)
    {
        OnUpdateFrame();

        fixed (GeometryPtr* geometries = geometryHandles)
        fixed (InstancePtr* instances = instanceHandles)
        {
            var error = Update(_context, geometries, bCount, instances, iCount);
            if (error != 0) throw new Exception($"{error}");
        }
    }

    private void MouseAction(Inputs inputs, MouseButton button)
    {
        if (!_mouseLeftButton.Pressed)
        {
            _camera?.MouseDown(button, inputs.MouseX, inputs.MouseY);
        }
        else if (MouseX != inputs.MouseX || MouseY != inputs.MouseY)
        {
            _camera?.MouseDragging(button, MouseX, MouseY);
        }
    }

    private void ReleaseUnmanagedResources()
    {
        Close(Window, Camera, _context);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        foreach (var body in _bodies)
        {
            body.Dispose();
        }
        ReleaseUnmanagedResources();
    }

    ~OptixWindow()
    {
        ReleaseUnmanagedResources();
    }
}