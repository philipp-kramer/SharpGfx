using System;
using SharpGfx.Host;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry
{
    public static class FoldedQuad
    {
        private static readonly Space Space = new HostSpace(Domain.Model);

        public static float[] GetVertices(Func<float, float, float> f)
        {
            return new[]{
                -0.5f, f(-0.5f,-0.5f),-0.5f,
                 0.5f, f( 0.5f,-0.5f),-0.5f,
                 0.5f, f( 0.5f, 0.5f), 0.5f,

                -0.5f, f(-0.5f,-0.5f),-0.5f,
                 0.5f, f( 0.5f, 0.5f), 0.5f,
                -0.5f, f(-0.5f, 0.5f), 0.5f,
            };
        }

        public static float[] GetNormals(Func<float, float, float> f)
        {
            var origin = Space.Vector3(-0.5f, f(-0.5f,-0.5f),-0.5f);
            var shared = Space.Vector3(0.5f, f( 0.5f, 0.5f), 0.5f);
            var t0 = Space.Vector3(0.5f, f( 0.5f,-0.5f),-0.5f);
            var t1 = Space.Vector3(-0.5f, f(-0.5f, 0.5f), 0.5f);

            var d = shared - origin;
            var n0 = IVector3.Cross(d, t0 - origin).Normalized();
            var n1 = IVector3.Cross(t1 - origin, d).Normalized();

            return new[] {
                n0[0],  n0[1], n0[2],
                n0[0],  n0[1], n0[2],
                n0[0],  n0[1], n0[2],

                n1[0],  n1[1], n1[2],
                n1[0],  n1[1], n1[2],
                n1[0],  n1[1], n1[2],
            };
        }
    }
}