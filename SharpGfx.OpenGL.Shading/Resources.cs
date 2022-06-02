using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace SharpGfx.OpenGL.Shading
{
    public static class Resources
    {
        /// <summary>
        /// Get a string resource.
        /// </summary>
        public static string GetSource(string name)
        {
            var assembly = Assembly.GetExecutingAssembly();
            using var stream = assembly.GetManifestResourceStream(GetFullPath(assembly, $"Sources.{name}"));
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        /// <summary>
        /// Construct the full resource path by pre-pending the name of the assembly containing type T and 'Resources'.
        /// Thus all resources must be put under folder named 'Resources' in the project structure.
        /// </summary>
        public static string GetFullPath(Assembly assembly, string name)
        {
            var resourceNames = assembly.GetManifestResourceNames();
            try
            {
                return resourceNames.Single(path => path == $"{assembly.GetName().Name}.{name}");
            }
            catch (InvalidOperationException)
            {
                var allResources = string.Join(Environment.NewLine, resourceNames);
                throw new InvalidOperationException($"resource '{name}' not found. Available resources:\n{allResources}");
            }
        }
    }
}