using System;
using System.Runtime.InteropServices;
using SharpGfx.Primitives;

namespace SharpGfx.OptiX;

internal class OptixInstance : Instance
{
    [DllImport(@".\optix.dll", EntryPoint = "Instance_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe InstancePtr CreateInstance(GeometryPtr geometry, float* transform);

    [DllImport(@".\optix.dll", EntryPoint = "Instance_Update", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe void Update(InstancePtr instance, float* transform);

    [DllImport(@".\optix.dll", EntryPoint = "Instance_Destroy", CallingConvention = CallingConvention.StdCall)]
    private static extern void DestroyInstance(InstancePtr instance);

    internal OptixBody Body { get; }
    private InstancePtr _handle;
    private readonly float[] _transformMatrix;

    internal OptixInstance(Space space, string name, OptixBody body)
        : base(space, name)
    {
        Body = body;
        _transformMatrix = new float[12];
        body.Use();
    }

    internal unsafe InstancePtr Create()
    {
        UpdateTransform();
        fixed (float* transform = _transformMatrix)
        {
            _handle = CreateInstance(Body.Handle, transform);
        }

        return _handle;
    }

    public override Instance Scale(float scale)
    {
        base.Scale(scale);
        if (!_handle.IsZero) Update();
        return this;
    }

    public override Instance Scale(IVector3 scale)
    {
        base.Scale(scale);
        if (!_handle.IsZero) Update();
        return this;
    }

    public override Instance Translate(IVector3 delta)
    {
        base.Translate(delta);
        if (!_handle.IsZero) Update();
        return this;
    }

    public override Instance RotateX(float angle)
    {
        base.RotateX(angle);
        if (!_handle.IsZero) Update();
        return this;
    }

    public override Instance RotateY(float angle)
    {
        base.RotateY(angle);
        if (!_handle.IsZero) Update();
        return this;
    }

    public override Instance RotateZ(float angle)
    {
        base.RotateZ(angle);
        if (!_handle.IsZero) Update();
        return this;
    }

    public override void Render()
    {
        throw new NotSupportedException();
    }

    private void UpdateTransform()
    {
        var elements = Transform.Elements;
        _transformMatrix[0] = elements[0, 0];
        _transformMatrix[1] = elements[1, 0];
        _transformMatrix[2] = elements[2, 0];
        _transformMatrix[3] = elements[3, 0];
        _transformMatrix[4] = elements[0, 1];
        _transformMatrix[5] = elements[1, 1];
        _transformMatrix[6] = elements[2, 1];
        _transformMatrix[7] = elements[3, 1];
        _transformMatrix[8] = elements[0, 2];
        _transformMatrix[9] = elements[1, 2];
        _transformMatrix[10] = elements[2, 2];
        _transformMatrix[11] = elements[3, 2];
    }

    private unsafe void Update()
    {
        UpdateTransform();
        fixed (float* transform = _transformMatrix)
        {
            Update(_handle, transform);
        }
    }
        
    private void ReleaseUnmanagedResources()
    {
        DestroyInstance(_handle);
    }

    protected override void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
        Body.Unuse();
        base.Dispose(disposing);
    }

    ~OptixInstance()
    {
        ReleaseUnmanagedResources();
    }
}