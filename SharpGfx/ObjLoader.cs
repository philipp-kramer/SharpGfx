using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using ObjLoader.Loader.Data.Elements;
using ObjLoader.Loader.Data.VertexData;
using ObjLoader.Loader.Loaders;

namespace SharpGfx
{
    public static class ObjLoader
    {
        private readonly struct Indices
        {
            internal readonly uint Normal;
            internal readonly uint Texture;

            public Indices(uint normal, uint texture)
            {
                Normal = normal;
                Texture = texture;
            }
        }

        private class MaterialStreamProvider : IMaterialStreamProvider
        {
            private readonly string _path;

            public MaterialStreamProvider(string path)
            {
                _path = path;
            }

            public Stream Open(string materialFilePath)
            {
                return new FileStream(Path.Combine(_path, materialFilePath), FileMode.Open);
            }
        }

        public static ICollection<RenderObject> Load(
            string path, 
            string filename, 
            Device device, 
            Func<TextureHandle, int, Material> createTextureMaterial,
            Func<Material> createPlainMaterial)
        {
            var factory = new ObjLoaderFactory();
            var loader = factory.Create(new MaterialStreamProvider(path));
            LoadResult obj;
            using (var fileStream = new FileStream(Path.Combine(path, filename), FileMode.Open))
            {
                obj = loader.Load(fileStream);
            }
            var textures = new Dictionary<int, Bitmap>();
            for (var index = 0; index < obj.Materials.Count; index++)
            {
                var material = obj.Materials[index];
                textures.Add(index, GetTexture(path, material.DiffuseTextureMap));
            }

            var dependencies = GetDependencies(obj);
            var vertices = GetComponents(obj.Vertices);
            var attributes = new List<(string, float[], int)>
            {
                ("positionIn", vertices, 3)
            };
            if (obj.Normals.Any())
            {
                var normals = GetComponents(obj.Normals, dependencies);
                attributes.Add(("normalIn", normals, 3));
            }

            if (obj.Textures.Any())
            {
                var coordinates = GetComponents(obj.Textures, dependencies);
                attributes.Add(("texCoordIn", coordinates, 2));
            }

            var scene = new List<RenderObject>();
            for (int group = 0; group < obj.Groups.Count; group++)
            {
                scene.Add(LoadGroup(device, createTextureMaterial, createPlainMaterial, obj, textures, group, attributes));
            }

            return scene;
        }

        private static Dictionary<uint, Indices> GetDependencies(LoadResult obj)
        {
            var dependencies = new Dictionary<uint, Indices>();
            foreach (var t in obj.Groups)
            {
                foreach (var face in t.Faces)
                {
                    for (int i = 0; i < face.Count; i++)
                    {
                        dependencies[GetVertexIndex(face, i)] = new Indices(GetNormalIndex(face, i), GetTextureIndex(face, i));
                    }
                }
            }

            return dependencies;
        }

        // TODO: rework ObjLoader and try using array in it directly (if length of array can be determined ahead of time)
        private static float[] GetComponents(IList<Vertex> vertices)
        {
            return vertices
                .SelectMany(v => new[] { v.X, v.Y, v.Z })
                .ToArray();
        }

        private static float[] GetComponents(IList<Normal> normals, Dictionary<uint, Indices> dependencies)
        {
            var components = new float[dependencies.Count * 3];
            foreach (var mapping in dependencies.OrderBy(dependency => dependency.Key))
            {
                components[3 * mapping.Key + 0] = normals[(int) mapping.Value.Normal].X;
                components[3 * mapping.Key + 1] = normals[(int) mapping.Value.Normal].Y;
                components[3 * mapping.Key + 2] = normals[(int) mapping.Value.Normal].Z;
            }
            return components;
        }

        private static float[] GetComponents(IList<Texture> textures, Dictionary<uint, Indices> dependencies)
        {
            var components = new float[dependencies.Count * 2];
            foreach (var mapping in dependencies.OrderBy(dependency => dependency.Key))
            {
                components[2 * mapping.Key + 0] = textures[(int)mapping.Value.Texture].X;
                components[2 * mapping.Key + 1] = 1 - textures[(int)mapping.Value.Texture].Y;
            }
            return components;
        }

        private static RenderObject LoadGroup(
            Device device,
            Func<TextureHandle, int, Material> createTextureMaterial,
            Func<Material> createPlainMaterial, 
            LoadResult obj,
            Dictionary<int, Bitmap> textures,
            int index,
            List<(string, float[], int)> attributes)
        {
            var group = obj.Groups[index];
            var groupMaterial = group.Material;
            int materialIndex = obj.Materials.IndexOf(groupMaterial);
            var material = groupMaterial != null && textures.TryGetValue(materialIndex, out var texture)
                ? createTextureMaterial(device.Texture(texture), materialIndex)
                : createPlainMaterial();
            return device
                .Object(
                    device.Model(),
                    material,
                    GetIndices(group),
                    attributes.ToArray());
        }

        private static Bitmap GetTexture(string path, string filename)
        {
            using var fileStream = new FileStream(Path.Combine(path, filename), FileMode.Open);
            return new Bitmap(fileStream);
        }

        private static uint[] GetIndices(Group group)
        {
            var indices = new List<uint>();
            foreach (var face in group.Faces)
            {
                if (3 <= face.Count && face.Count <= 4)
                {
                    uint vertexIndex0 = GetVertexIndex(face, 0);
                    indices.Add(vertexIndex0);

                    indices.Add(GetVertexIndex(face, 1));

                    uint vertexIndex2 = GetVertexIndex(face, 2);
                    indices.Add(vertexIndex2);

                    if (face.Count == 4)
                    {
                        indices.Add(vertexIndex0);
                        indices.Add(vertexIndex2);
                        indices.Add(GetVertexIndex(face, 3));
                    }
                }
                else
                {
                    throw new NotSupportedException("faces having other than 3 or 4 corners are not supported");
                }
            }

            return indices.ToArray();
        }

        private static uint GetVertexIndex(Face face, int index)
        {
            return (uint) face[index].VertexIndex - 1;
        }

        private static uint GetNormalIndex(Face face, int index)
        {
            return (uint) face[index].NormalIndex - 1;
        }

        private static uint GetTextureIndex(Face face, int index)
        {
            return (uint) face[index].TextureIndex - 1;
        }

    }
}
