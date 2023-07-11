using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace SharpGfx.OptiX;

internal class OptixBody : Body
{
    [DllImport(@".\optix.dll", EntryPoint = "Geometry_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe GeometryPtr Create(ContextPtr context, float* positions, float* texPositions, int verticesCount);

    [DllImport(@".\optix.dll", EntryPoint = "IntIndexedGeometry_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe GeometryPtr Create(ContextPtr context, float* positions, float* texPositions, int verticesCount, uint* indices, int triangleCount);

    [DllImport(@".\optix.dll", EntryPoint = "ShortIndexedGeometry_Create", CallingConvention = CallingConvention.StdCall)]
    private static extern unsafe GeometryPtr Create(ContextPtr context, float* positions, float* texPositions, int verticesCount, ushort* indices, int triangleCount);


    [DllImport(@".\optix.dll", EntryPoint = "Geometry_Destroy", CallingConvention = CallingConvention.StdCall)]
    private static extern void Dispose(GeometryPtr geometry);

    internal GeometryPtr Handle { get; }

    internal unsafe OptixBody(ContextPtr context, Material material, params IVertexAttribute[] attributes)
        : base(material, GetVertexCount(attributes, 0))
    {
        fixed (float* positions = GetPositions(attributes))
        fixed (float* texPosition = GetTexPositions(attributes))
        {
            Handle = Create(context, positions, texPosition, VertexCount);
        }
    }

    internal unsafe OptixBody(ContextPtr context, Material material, uint[] indices, params IVertexAttribute[] attributes)
        : base(material, GetVertexCount(attributes, 0))
    {
        if (indices.Length % 3 != 0) throw new ArgumentException(nameof(indices));

        fixed (float* positions = GetPositions(attributes))
        fixed (float* texPositions = GetTexPositions(attributes))
        fixed (uint* fixedIndices = indices)
        {
            Handle = Create(context, positions, texPositions, VertexCount, fixedIndices, indices.Length / 3);
        }
    }

    internal unsafe OptixBody(ContextPtr context, Material material, ushort[] indices, params IVertexAttribute[] attributes)
        : base(material, GetVertexCount(attributes, 0))
    {
        fixed (float* positions = GetPositions(attributes))
        fixed (float* texPositions = GetTexPositions(attributes))
        fixed (ushort* fixedIndices = indices)
        {
            Handle = Create(context, positions, texPositions, VertexCount, fixedIndices, GetTriangleCount(indices));
        }
    }

    protected override void Dispose(bool disposing)
    {
        Dispose(Handle);
    }

    private static float[] GetPositions(IVertexAttribute[] attributes)
    {
        return attributes
            .OfType<PositionVa>()
            .Single()
            .Values;
    }

    private static float[] GetTexPositions(IVertexAttribute[] attributes)
    {
        var result = attributes
            .OfType<TexPositionVa>()
            .ToArray();
        return result.Length switch
        {
            1 => result[0].Values,
            0 => null,
            _ => throw new ArgumentException(nameof(attributes))
        };
    }

    private static int GetVertexCount(IVertexAttribute[] attributes, int i)
    {
        if (attributes[i].Values.Length % attributes[i].Rank != 0) throw new InvalidOperationException();

        return attributes[i].Values.Length / attributes[i].Rank;
    }

    private static int GetTriangleCount(ushort[] indices)
    {
        if (indices.Length % 3 != 0) throw new ArgumentException(nameof(indices));

        return indices.Length / 3;
    }
}