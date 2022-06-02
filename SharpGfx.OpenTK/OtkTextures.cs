using System;
using System.Drawing;
using System.Drawing.Imaging;
using OpenTK.Graphics.OpenGL;
using SharpGfx.Primitives;
using PixelFormat = OpenTK.Graphics.OpenGL.PixelFormat;

namespace SharpGfx.OpenTK
{
    internal static class OtkTextures
    {
        public static int CreateTexture(IVector2 pixels, PixelInternalFormat pixelInternalFormat, PixelFormat pixelFormat, PixelType pixelType)
        {
            int texture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, texture);

            GL.TexImage2D(TextureTarget.Texture2D, 0, pixelInternalFormat, (int) pixels.X, (int) pixels.Y, 0, pixelFormat, pixelType, default);

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

        internal static TextureUnit GetTextureUnit(int unit)
        {
            var enumEntryName = nameof(TextureUnit.Texture0).Replace("0", unit.ToString());
            if (!Enum.TryParse(enumEntryName, out TextureUnit textureUnit)) throw new ArgumentOutOfRangeException(nameof(unit));
            return textureUnit;
        }
    }
}
