using System;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal static class OtkBuffer
    {
        internal static int CreateBuffer<T>(T[] data, BufferTarget target)
            where T : struct
        {
            int sizeOfT = Marshal.SizeOf<T>();

            int handle = GL.GenBuffer();
            GL.BindBuffer(target, handle);
            GL.BufferData(target, data.Length * Marshal.SizeOf<T>(), data, BufferUsageHint.StaticDraw);

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