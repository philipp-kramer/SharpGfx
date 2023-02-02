using System;
using System.Runtime.InteropServices;

namespace SharpGfx.OpenGL;

internal static class GlBuffer
{
    internal static uint CreateBuffer<T>(GlApi gl, T[] data, GlBufferTarget target)
        where T : struct
    {
        int sizeOfT = typeof(T).Name switch
        {
            "Double" => 8,
            "Single" => 4,
            "Int64" => 8,
            "UInt64" => 8,
            "Int32" => 4,
            "UInt32" => 4,
            "Int16" => 2,
            "UInt16" => 2,
            "Int8" => 1,
            "UInt8" => 1,
            _ => throw new NotSupportedException("only numeric types allowed"),
        };

        uint handle = gl.GenBuffer();
        gl.BindBuffer(target, handle);
        var hostHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            gl.BufferData(target, data.Length * sizeOfT, hostHandle.AddrOfPinnedObject());
        }
        finally
        {
            hostHandle.Free();
        }

        long bufferSize = gl.GetBufferParameter(target, GlBufferParameterName.BufferSize);
        if (data.Length * sizeOfT != bufferSize)
        {
            throw new ApplicationException("array not uploaded correctly");
        }

        gl.BindBuffer(target, 0);

        return handle;
    }
}