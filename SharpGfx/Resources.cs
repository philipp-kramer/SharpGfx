using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;

namespace SharpGfx
{
    public static class Resources
    {
        /// <summary>
        /// Get a string resource.
        /// </summary>
        public static string Get<T>(string path)
        {
            var fullPath = GetFullPath<T>(path);
            using var stream = Assembly.GetAssembly(typeof(T)).GetManifestResourceStream(fullPath);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        /// <summary>
        /// Get a bitmap resource. The bitmap must be disposed.
        /// </summary>
        public static Bitmap GetBitmap<T>(string path)
        {
            var resourcePath = GetFullPath<T>(path);
            using Stream resourceStream = Assembly.GetAssembly(typeof(T)).GetManifestResourceStream(resourcePath);
            return new Bitmap(resourceStream);
        }

        /// <summary>
        /// Construct the full resource path by pre-pending the name of the assembly containing type T and 'Resources'.
        /// Thus all resources must be put under folder named 'Resources' in the project structure.
        /// </summary>
        public static string GetFullPath<T>(string resourcePath)
        {
            var assembly = Assembly.GetAssembly(typeof(T));
            return assembly
                .GetManifestResourceNames()
                .Single(path => path == $"{assembly.GetName().Name}.{nameof(Resources)}.{resourcePath}");
        }
    }
}