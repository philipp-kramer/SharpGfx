using System;
using System.Collections.Generic;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry
{
    public static class Triangulation
    {
        /// <param name="space"></param>
        /// <param name="planePoint">point of cutting plane</param>
        /// <param name="planeNormal">normal of cutting plane</param>
        /// <param name="vertices">extended with additional vertices</param>
        /// <param name="normals">normalized normals, extended with additional normals</param>
        /// <param name="triangles">unchanged</param>
        /// <returns></returns>
        public static (List<uint>, List<uint>) Cut(
            Space space, 
            Point3 planePoint, 
            IVector3 planeNormal, 
            List<float> vertices, 
            List<float> normals, 
            List<uint> triangles)
        {
            var lower = new List<uint>();
            var upper = new List<uint>();

            for (int i = 0; i < triangles.Count; i += 3)
            {
                uint ia = triangles[i];
                uint ib = triangles[i + 1];
                uint ic = triangles[i + 2];
                var va = GetVertex(space, vertices, ia);
                var vb = GetVertex(space, vertices, ib);
                var vc = GetVertex(space, vertices, ic);

                uint cia = uint.MaxValue;
                uint cib = uint.MaxValue;
                uint cic = uint.MaxValue;
                float t;
                Point3 cva = default;
                Point3 cvb = default;
                Point3 cvc = default;
                if (TryIntersect(planePoint, planeNormal, va, vb, out t)) (cia, cva) = Interpolate(space, planeNormal, vertices, normals, ia, va, ib, vb, t);
                if (TryIntersect(planePoint, planeNormal, vb, vc, out t)) (cib, cvb) = Interpolate(space, planeNormal, vertices, normals, ib, vb, ic, vc, t);
                if (TryIntersect(planePoint, planeNormal, vc, va, out t)) (cic, cvc) = Interpolate(space, planeNormal, vertices, normals, ic, vc, ia, va, t);

                if (cib != uint.MaxValue && cic != uint.MaxValue) // corner c is cut
                {
                    bool side = planeNormal.Dot(cvc - vc) > 0;
                    AddTriangle(side ? lower : upper, ic, cic, cib);
                    AddQuad(side ? upper : lower, (va, ia), (vb, ib), (cvb, cib), (cvc, cic));
                }
                else if (cic != uint.MaxValue && cia != uint.MaxValue) // corner a is cut
                {
                    bool side = planeNormal.Dot(cva - va) > 0;
                    AddTriangle(side ? lower : upper, ia, cia, cic);
                    AddQuad(side ? upper : lower, (vb, ib), (vc, ic), (cvc, cic), (cva, cia));
                }
                else  if (cia != uint.MaxValue && cib != uint.MaxValue) // corner b is cut
                {
                    bool side = planeNormal.Dot(cvb - vb) > 0;
                    AddTriangle(side ? lower : upper, ib, cib, cia);
                    AddQuad(side ? upper : lower, (vc, ic), (va, ia), (cva, cia), (cvb, cib));
                }
                else
                {
                    bool side = planeNormal.Dot(planePoint - Point3.Center(va, vb, vc)) > 0;
                    AddTriangle(side ? lower : upper, ia, ib, ic);
                }
            }

            return (lower, upper);
        }

        private static (uint, Point3) Interpolate(
            Space space, 
            IVector3 planeNormal, 
            List<float> vertices,
            List<float> normals, 
            uint iStart, Point3 vStart,
            uint iEnd, Point3 vEnd, 
            float t)
        {
            var nStart = GetVertex(space, normals, iStart).Vector;
            var nEnd = GetVertex(space, normals, iEnd).Vector;
            var n = nStart + t * (nEnd - nStart);
            float cos = nStart.Dot(nEnd);
            float sin = MathF.Sqrt(1 - cos * cos);
            var delta = vEnd - vStart;
            float edge = (0.5f - MathF.Abs(t - 0.5f)) * delta.Length; // centering and length
            float side = MathF.Sign(delta.Dot(nEnd - nStart));
            float correction = 0.5f * edge * side * sin;
            var projectedN = n - n.Dot(planeNormal) * planeNormal;
            var vertex = vStart + t * (vEnd - vStart) + correction * projectedN;

            var index = AddVertex(vertices, vertex);
            AddVertex(normals, space.Origin3() + n);
            return (index, vertex);
        }

        private static void AddQuad(List<uint> triangles, (Point3, uint) corner1, (Point3, uint) corner2, (Point3, uint) cut1, (Point3, uint) cut2)
        {
            var delta1 = cut1.Item1 - corner1.Item1;
            var delta2 = cut2.Item1 - corner2.Item1;

            if (delta1.Dot(delta1) < delta2.Dot(delta2))
            {
                AddTriangle(triangles, corner1.Item2, corner2.Item2, cut1.Item2);
                AddTriangle(triangles, cut1.Item2, cut2.Item2, corner1.Item2);
            }
            else
            {
                AddTriangle(triangles, corner1.Item2, corner2.Item2, cut2.Item2);
                AddTriangle(triangles, cut1.Item2, cut2.Item2, corner2.Item2);
            }
        }

        private static void AddTriangle(List<uint> triangles, uint ia, uint ib, uint ic)
        {
            triangles.Add(ia);
            triangles.Add(ib);
            triangles.Add(ic);
        }

        private static Point3 GetVertex(Space space, List<float> vertices, uint vertex)
        {
            int index = 3 * (int) vertex;
            return space.Point3(vertices[index], vertices[index + 1], vertices[index + 2]);
        }

        private static uint AddVertex(List<float> vertices, Point3 intersection)
        {
            var offset = (uint) vertices.Count / 3;
            vertices.Add(intersection.X);
            vertices.Add(intersection.Y);
            vertices.Add(intersection.Z);
            return offset;
        }

        private static bool TryIntersect(Point3 planePoint, IVector3 planeNormal, Point3 start, Point3 end, out float t)
        {
            t = planeNormal.Dot(planePoint - start) / planeNormal.Dot(end - start);
            return 0 < t && t < 1;
        }
    }
}
