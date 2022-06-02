using System;
using System.Runtime.InteropServices;

namespace SharpGfx.OpenGL
{
    internal static class OglBuffer
    {
        internal static uint CreateBuffer<T>(T[] data, GlBufferTarget target)
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

            uint handle = GL.GenBuffer();
            GL.BindBuffer(target, handle);
            unsafe
            {
                GL.BufferData(target, data.Length * sizeOfT, Marshal.UnsafeAddrOfPinnedArrayElement(data, 0).ToPointer());
            }

            long bufferSize = GL.GetBufferParameter(target, GlBufferParameterName.BufferSize);
            if (data.Length * sizeOfT != bufferSize)
            {
                throw new ApplicationException("array not uploaded correctly");
            }

            GL.BindBuffer(target, 0);

            return handle;
        }
    }
}