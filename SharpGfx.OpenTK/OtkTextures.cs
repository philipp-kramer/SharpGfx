using System;
using System.Drawing;
using System.Drawing.Imaging;
using OpenTK.Graphics.OpenGL;
using PixelFormat = OpenTK.Graphics.OpenGL.PixelFormat;

namespace SharpGfx.OpenTK
{
    internal static class OtkTextures
    {
        public static int CreateTexture(Size pixels, PixelInternalFormat pixelInternalFormat, PixelFormat pixelFormat, PixelType pixelType)
        {
            int texture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, texture);

            GL.TexImage2D(TextureTarget.Texture2D, 0, pixelInternalFormat, pixels.Width, pixels.Height, 0, pixelFormat, pixelType, default);

            int linear = (int)TextureMinFilter.Linear;
            GL.TexParameterI(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, ref linear);
            GL.TexParameterI(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, ref linear);

            return texture;
        }

        internal static int CreateAutoMipmapTexture(Bitmap bitmap)
        {
            int handle = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, handle);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int) TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int) TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int) TextureWrapMode.Repeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int) TextureWrapMode.Repeat);

            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                try
                {
                GL.TexImage2D(
                    TextureTarget.Texture2D,
                    0,
                    PixelInternalFormat.Rgba,
                    bitmap.Width,
                    bitmap.Height,
                    0,
                    PixelFormat.Bgra,
                    PixelType.UnsignedByte,
                    bitmapData.Scan0);
            }
            finally
            {
                bitmap.UnlockBits(bitmapData);

            }

            GL.BindTexture(TextureTarget.Texture2D, 0);
            return handle;
        }

        internal static int CreateMipmapTexture(Bitmap bitmap)
        {
            GL.CreateTextures(TextureTarget.Texture2D, 1, out int texture);
            GL.BindTexture(TextureTarget.Texture2D, texture);
            int levels = (int)Math.Min(
                Math.Log(bitmap.Width, 2),
                Math.Log(bitmap.Height, 2));
            GL.TextureStorage2D(
                texture,
                levels,
                SizedInternalFormat.Rgba32f,
                bitmap.Width,
                bitmap.Height);

            int bWidth = bitmap.Width;
            int bHeight = bitmap.Height;
            var below = ExtractMipmap(bitmap, bWidth, bHeight);
            GL.TextureSubImage2D(texture, 0, 0, 0, bWidth, bHeight, PixelFormat.Rgba, PixelType.Float, below);
            for (int level = 1; level < levels; level++)
            {
                int aWidth = bWidth / 2;
                int aHeight = bHeight / 2;
                var above = GetMipmapAbove(below, bWidth, aWidth, aHeight);
                GL.TextureSubImage2D(texture, level, 0, 0, aWidth, aHeight, PixelFormat.Rgba, PixelType.Float, above);

                below = above;
                bWidth = aWidth;
                bHeight = aHeight;
            }

            var textureMinFilter = (int) All.Nearest; // All.LinearMipmapNearest; LinearMipmapLinear
            var textureMagFilter = (int) All.Linear;
            GL.TextureParameterI(texture, TextureParameterName.TextureMinFilter, ref textureMinFilter);
            GL.TextureParameterI(texture, TextureParameterName.TextureMagFilter, ref textureMagFilter);
            // data not needed from here on, OpenGL has the data
            return texture;
        }

        private static float[] ExtractMipmap(Bitmap bitmap, int width, int height)
        {
            var pixels = new float[width * height * 4];
            int index = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pixel = bitmap.GetPixel(x, y);
                    pixels[index++] = pixel.R / 255f;
                    pixels[index++] = pixel.G / 255f;
                    pixels[index++] = pixel.B / 255f;
                    pixels[index++] = pixel.A / 255f;
                }
            }

            return pixels;
        }

        // könnte mit Gauss-Filter verbessert werden
        private static float[] GetMipmapAbove(float[] below, int bWidth, int aWidth, int aHeight)
        {
            var above = new float[aHeight * aWidth *4];
            int a = 0;
            for (int y = 0; y < aHeight; y++)
            {
                int b = 2 * y * 4 * bWidth;
                for (int x = 0; x < aWidth; x++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        above[a] = below[b] +
                                   below[b + 4] +
                                   below[b + 4 * bWidth] +
                                   below[b + 4 * (bWidth + 1)];
                        above[a] *= 1f/4f;
                        a++; // next channel
                        b++;
                    }

                    b += 4; // skip pixel
                }
            }

            return above;
        }

        public static void ActivateTexture(TextureHandle handle, int unit)
        {
            var textureUnit = GetTextureUnit(unit);
            GL.ActiveTexture(textureUnit);
            GL.BindTexture(TextureTarget.Texture2D, ((OtkTextureHandle)handle)?.Handle ?? 0);
        }

        public static void ClearTexture(int unit)
        {
            var textureUnit = GetTextureUnit(unit);
            GL.ActiveTexture(textureUnit);
            GL.BindTexture(TextureTarget.Texture2D, 0);
        }

        public static void DeleteTexture(TextureHandle handle)
        {
            if (handle != null)
            {
                GL.DeleteTexture(((OtkTextureHandle)handle).Handle);
            }
        }

        private static TextureUnit GetTextureUnit(int unit)
        {
            var enumEntryName = nameof(TextureUnit.Texture0).Replace("0", unit.ToString());
            if (!Enum.TryParse(enumEntryName, out TextureUnit textureUnit)) throw new ArgumentOutOfRangeException(nameof(unit));
            return textureUnit;
        }
    }
}
