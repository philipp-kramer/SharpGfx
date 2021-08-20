using System;
using System.Collections.Generic;
using System.Linq;
using SharpGfx.Host;
using SharpGfx.Primitives;

namespace SharpGfx.Geometry
{
    public static class Tree
    {
        private class Branch
        {
            internal readonly float Length;
            internal readonly int Leaves;
            internal readonly Branch[] Branches;

            public Branch(float length, int leaves, Branch[] branches)
            {
                Length = length;
                Leaves = leaves;
                Branches = branches;
            }
        }

        private readonly struct Crotch
        {
            internal readonly Point3 Start;
            internal readonly float Radius;
            internal readonly float PitchAngle;
            internal readonly float YawAngle;
            internal readonly Branch Structure;

            internal Crotch(Point3 start, float radius, float pitchAngle, float yawAngle, Branch structure)
            {
                Start = start;
                Radius = radius;
                PitchAngle = pitchAngle;
                YawAngle = yawAngle;
                Structure = structure;
            }
            
	        internal Point3 Step(float distance)
	        {
	            var step = Space.Vector4(distance, 0, 0, 1) *
                           Space.RotationZ4(PitchAngle) *
	                       Space.RotationY4(YawAngle);
	            return Start + step.Xyz;
	        }
        }

        private static readonly Space Space = new HostSpace(Domain.Model);
        private static readonly Random Random = new Random();
        private static readonly float UpwardsAngle = MathF.PI / 2;

        public static void Get(
            int leaves,
            float thickness,
            out List<float> vertices,
            out List<float> normals,
            out List<float> texture,
            out List<ushort> indices,
            out List<float> leavesVertices,
            out List<float> leavesNormals)
        {
            var structure = BuildBranchStructure(leaves, leaves);
            float maxLength = GetMaxLength(structure);
            float scale = 0.5f / maxLength;

            float stemRadius = 1.25f * GetRadius(structure, thickness, scale);
            var root = new Crotch(Space.Origin3(), stemRadius, UpwardsAngle, 0, structure);

            vertices = new List<float>();
            normals = new List<float>();
            indices = new List<ushort>();
            texture = new List<float>();
            leavesVertices = new List<float>();
            leavesNormals = new List<float>();
            BuildGeometry(root, thickness, vertices, normals, texture, indices, leavesVertices, leavesNormals, scale);
        }

        private static Branch BuildBranchStructure(int leaves, int totalLeaves)
        {
            if (leaves > 2)
            {
                var r = 0.5f * ((float)Random.NextDouble() + 0.5f);
                int split = (int)(Weighted((float)leaves / totalLeaves, r, 1-r) * leaves);
                return new Branch(
                    1 + MathF.Log(leaves),
                    leaves,
                    new[]
                    {
                        BuildBranchStructure(split, totalLeaves),
                        BuildBranchStructure(leaves - split, totalLeaves)
                    });
            }
            else
            {
                return new Branch(2, leaves, null);
            }
        }

        private static float GetMaxLength(Branch branch)
        {
            return branch.Length + (branch.Branches?.Select(GetMaxLength).Max() ?? 0);
        }

        private static void BuildGeometry(
            Crotch root,
            float thickness,
            List<float> vertices,
            List<float> normals,
            List<float> texture,
            List<ushort> indices,
            List<float> leavesVertices,
            List<float> leavesNormals,
            float scale)
        {
            var queue = new Queue<Crotch>();
            queue.Enqueue(root);

            while (queue.Any())
            {
                var crotch = queue.Dequeue();

                var end = crotch.Step(crotch.Structure.Length * scale);

                const int corners = 5;
                const int layers = 1;
                float radius = GetRadius(crotch.Structure, thickness, scale);
                float Radius(int layer) => layer == 0 ? crotch.Radius : radius;
                Point3 Center(float layer)
                {
                    return Point3.Combine(layer, end, crotch.Start);
                }
                var branchIndices = Cylinder.GetIndices(corners, layers);
                ShiftIndices(branchIndices, (ushort)vertices.Count);
                indices.AddRange(branchIndices);

                vertices.AddRange(Cylinder.GetVertices(Space, corners, layers, Radius, Center, out var nrs));
                normals.AddRange(nrs);
                texture.AddRange(Cylinder.GetTextureVertices(corners, layers));

                AddLeafPairs(leavesVertices, leavesNormals, Point3.Combine(0.5f, crotch.Start, end), crotch.YawAngle);
                AddLeafPairs(leavesVertices, leavesNormals, end, crotch.YawAngle);

                var branches = crotch.Structure.Branches;
                if (branches != null)
                {
                    queue.Enqueue(GrowCrotch(crotch, end, radius, branches[0], MathF.PI / 4));
                    queue.Enqueue(GrowCrotch(crotch, end, radius, branches[1], -MathF.PI / 4));
                }
            }
        }

        private static float GetRadius(Branch structure, float thickness, float scale)
        {
            return thickness * scale * MathF.Sqrt(structure.Leaves);
        }

        private static Crotch GrowCrotch(Crotch crotch, Point3 end, float radius, Branch branch, float yawOpening)
        {
            var weight = (float)branch.Leaves / crotch.Structure.Leaves;
            var adjustedPitch = Weighted(weight, UpwardsAngle, crotch.PitchAngle + MathF.PI / 4);
            var adjustedYaw = crotch.YawAngle + yawOpening;
            return new Crotch(end, radius, adjustedPitch, adjustedYaw, branch);
        }

        private static float Weighted(float wa, float a, float b)
        {
            return wa * a + (1f - wa) * b;
        }

        private static void AddLeafPairs(List<float> leavesVertices, List<float> leavesNormals, Point3 end, float yawAngle)
        {
            var yaw = Space.Rotation2(yawAngle - MathF.PI / 4);
            var rotation = Space.Matrix4(
                yaw[0, 0], 0, yaw[0, 1], 0,
                0, 1, 0, 0,
                yaw[1, 0], 0, yaw[1, 1], 0,
                0, 0, 0, 1);
            var delta = (rotation * Space.Vector4(0.5f, 0, 0.5f, 1)).Xyz;
            AddLeaves(leavesVertices, leavesNormals, end, rotation, 1, delta);
            AddLeaves(leavesVertices, leavesNormals, end, rotation, -1, delta);
        }

        private static void AddLeaves(
            List<float> leavesVertices,
            List<float> leavesNormals,
            Point3 end,
            Matrix4 rotation,
            int upSign,
            Vector3 delta)
        {
            const float slope = 0.25f;
            const float scale = 0.01f;

            Func<float, float, float> SlopeFunc() => (x, z) => slope * Math.Max(upSign * x, upSign * z);
            var vertices = Quad.GetFlatVertices(SlopeFunc());
            var normals = Quad.GetFlatNormals(SlopeFunc());

            for (int i = 0; i < vertices.Length; i += 3)
            {
                var vector = GetVector3H(vertices, i);
                var rotated = (rotation * vector).Xyz;
                var vertex = end + (rotated + delta * upSign) * scale;
                leavesVertices.Add(vertex.X);
                leavesVertices.Add(vertex.Y);
                leavesVertices.Add(vertex.Z);
                leavesNormals.Add(normals[i + 0]);
                leavesNormals.Add(normals[i + 1]);
                leavesNormals.Add(normals[i + 2]);
            }
        }

        private static Vector4 GetVector3H(float[] values, long offset)
        {
            return Space.Vector4(
                values[offset + 0],
                values[offset + 1],
                values[offset + 2],
                1);
        }

        private static void ShiftIndices(ushort[] branchIndices, int verticesCount)
        {
            for (var i = 0; i < branchIndices.Length; i++)
            {
                branchIndices[i] += (ushort)(verticesCount / 3);
            }
        }
    }
}
