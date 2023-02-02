using System;
using SharpGfx.Primitives;

namespace SharpGfx.Host;

public readonly struct HostMatrix4 : Matrix4
{
    private readonly Space _space;

    public HostMatrix4(Space space, float[,] elements)
    {
        _space = space;
        Elements = elements;
    }

    Space IPrimitive.Space => _space;
    public float[,] Elements { get; }
    public float this[int row, int col]
    {
        get => Elements[row, col];
        set => Elements[row, col] = value;
    }

    public Matrix4 ToSpace(Space space)
    {
        return new HostMatrix4(space, Elements);
    }

    public IVector4 Mul(IVector4 r)
    {
        return new HostVector4(
            r.Space,
            Elements[0, 0] * r.X + Elements[0, 1] * r.Y + Elements[0, 2] * r.Z + Elements[0, 3] * r.W,
            Elements[1, 0] * r.X + Elements[1, 1] * r.Y + Elements[1, 2] * r.Z + Elements[1, 3] * r.W,
            Elements[2, 0] * r.X + Elements[2, 1] * r.Y + Elements[2, 2] * r.Z + Elements[2, 3] * r.W,
            Elements[3, 0] * r.X + Elements[3, 1] * r.Y + Elements[3, 2] * r.Z + Elements[3, 3] * r.W);
    }

    public Matrix4 Mul(Matrix4 r)
    {
        float[,] result = new float[4, 4];

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    sum += Elements[row, k] * r[k, col];
                }
                result[row, col] = sum;
            }
        }

        return new HostMatrix4(r.Space, result);
    }

    public Matrix4 Transposed()
    {
        var result = new[,]
        {
            {Elements[0, 0], Elements[1, 0], Elements[2, 0], Elements[3, 0] },
            {Elements[0, 1], Elements[1, 1], Elements[2, 1], Elements[3, 1] },
            {Elements[0, 2], Elements[1, 2], Elements[2, 2], Elements[3, 2] },
            {Elements[0, 3], Elements[1, 3], Elements[2, 3], Elements[3, 3] },
        };
        return new HostMatrix4(_space, result);
    }

    public static Matrix4 GetView(Space space, HostVector3 eye, HostVector3 lookAt, HostVector3 up)
    {
        var right = lookAt
            .Cross(up)
            .Normalized();
        var orthoUp = right.Cross(lookAt);
        Matrix4 transI = new HostMatrix4(
            space,
            new[,] {
                { 1, 0, 0, 0},
                { 0, 1, 0, 0},
                { 0, 0, 1, 0},
                { -eye.X, -eye.Y, -eye.Z, 1}
            });
        // negate lookAt to switch handedness
        Matrix4 rotI = new HostMatrix4(
            space,
            new[,] {
                { right.X, orthoUp.X, -lookAt.X, 0},
                { right.Y, orthoUp.Y, -lookAt.Y, 0},
                { right.Z, orthoUp.Z, -lookAt.Z, 0},
                { 0, 0, 0, 1}
            });
        return transI * rotI;
    }

    public static Matrix4 GetProjection(Space space, float fovy, float aspect, float near, float far)
    {
        float scale = 1 / MathF.Tan(0.5f * fovy);
        return new HostMatrix4(space, new float[4, 4])
        {
            [0, 0] = scale / aspect, // scale the x coordinates of the projected point 
            [1, 1] = scale, // scale the y coordinates of the projected point 
            [2, 2] = (near + far) / (near - far), // remap z to [0,1] 
            [3, 2] = 2 * near * far / (near - far), // remap z [0,1] 
            [2, 3] = -1 // set w = -z 
        };
    }

    public static Matrix4 GetProjection(Space space, float left, float right, float bottom, float top, float near, float far)
    {
        return new HostMatrix4(space, new float[4, 4])
        {
            [0, 0] = 2 * near / (right - left), 
            [1, 1] = 2 * near / (top - bottom),
            [2, 2] = -(far + near) / (far - near), // remap z to [0,1] 
            [3, 2] = -2 * near * far / (far - near), // remap z [0,1] 
            [2, 3] = -1 // set w = -z 
        };
    }
}