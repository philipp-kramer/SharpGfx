using System;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal static class OtkBuffer
    {
        internal static int CreateBuffer<T>(T[] data, BufferTarget target)
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

            int handle = GL.GenBuffer();
            GL.BindBuffer(target, handle);
            GL.BufferData(target, data.Length * sizeOfT, data, BufferUsageHint.StaticDraw);

            GL.GetBufferParameter(target, BufferParameterName.BufferSize, out int bufferSize);
            if (data.Length * sizeOfT != bufferSize)
            {
                throw new ApplicationException("array not uploaded correctly");
            }

            GL.BindBuffer(target, 0);
            return handle;
        }
    }
}