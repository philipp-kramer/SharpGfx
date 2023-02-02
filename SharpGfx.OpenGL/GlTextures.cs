using System;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpGfx.OpenGL;

internal static class GlTextures
{
    public static uint CreateTexture(GlApi gl, int width, int height, GlPixelFormat pixelFormat, GlPixelType pixelType)
    {
        uint texture = gl.GenTexture();
        gl.BindTexture(GlTextureTarget.Texture2D, texture);

        gl.TextureParameterI(GlTextureTarget.Texture2D, GlTextureParameterName.MinFilter, GlTextureParameter.Linear);
        gl.TextureParameterI(GlTextureTarget.Texture2D, GlTextureParameterName.MagFilter, GlTextureParameter.Linear);

        gl.TexImage2D(GlTextureTarget.Texture2D, 0, width, height, 0, pixelFormat, pixelType, IntPtr.Zero);

        return texture;
    }

    internal static uint CreateMipmapTexture(GlApi gl, Image<Rgba32> image)
    {
        uint texture = gl.GenTexture();
        gl.BindTexture(GlTextureTarget.Texture2D, texture);
        gl.TextureParameterI(GlTextureTarget.Texture2D, GlTextureParameterName.MinFilter, GlTextureParameter.Linear);
        gl.TextureParameterI(GlTextureTarget.Texture2D, GlTextureParameterName.MagFilter, GlTextureParameter.Linear);

        gl.TextureParameterI(GlTextureTarget.Texture2D, GlTextureParameterName.WrapS, GlTextureParameter.Repeat);
        gl.TextureParameterI(GlTextureTarget.Texture2D, GlTextureParameterName.WrapT, GlTextureParameter.Repeat);

        gl.GenerateMipmap(GlTextureTarget.Texture2D);

        var pixels = new byte[image.Width * image.Height * 4];
        image.CopyPixelDataTo(pixels);
        var handle = GCHandle.Alloc(pixels, GCHandleType.Pinned);
        try
        {
            gl.TexImage2D(
                GlTextureTarget.Texture2D,
                0,
                image.Width,
                image.Height,
                0,
                GlPixelFormat.Rgba,
                GlPixelType.UnsignedByte,
                handle.AddrOfPinnedObject());

            gl.BindTexture(GlTextureTarget.Texture2D, 0);
        }
        finally
        {
            handle.Free();
        }

        return texture;
    }

    // do it ourselves
    //internal static unsafe uint CreateMipmapTexture(Bitmap bitmap)
    //{
    //    uint texture = GL.CreateTexture(GlTextureTarget.Texture2D);
    //    GL.BindTexture(GlTextureTarget.Texture2D, texture);
    //    int levels = (int)Math.Min(
    //        Math.Log(bitmap.Width, 2),
    //        Math.Log(bitmap.Height, 2));
    //    GL.TextureStorage2D(
    //        texture,
    //        levels,
    //        GlSizedInternalFormat.Rgba32f,
    //        bitmap.Width,
    //        bitmap.Height);

    //    int bWidth = bitmap.Width;
    //    int bHeight = bitmap.Height;

    //    var below = new float[bWidth * bHeight * 4];
    //    ExtractMipmap(bitmap, below, bWidth, bHeight);
    //    fixed (float* pixels = below)
    //    {
    //        GL.TextureSubImage2D(texture, 0, 0, 0, bWidth, bHeight, GlPixelFormat.Rgba, GlPixelType.Float, pixels);
    //    }

    //    for (int level = 1; level < levels; level++)
    //    {
    //        int aWidth = bWidth / 2;
    //        int aHeight = bHeight / 2;
    //        var above = new float[aHeight * aWidth * 4];
    //        GetMipmapAbove(below, above, bWidth, aWidth, aHeight);
    //        fixed (float* pixels = new float[aHeight * aWidth * 4])
    //        {
    //            GL.TextureSubImage2D(texture, level, 0, 0, aWidth, aHeight, GlPixelFormat.Rgba, GlPixelType.Float, pixels);
    //        }

    //        below = above;
    //        bWidth = aWidth;
    //        bHeight = aHeight;
    //    }

    //    var textureMinFilter = GlTextureParameter.Nearest; // LinearMipmapNearest, LinearMipmapLinear
    //    var textureMagFilter = GlTextureParameter.Linear;
    //    GL.TextureParameterI(texture, GlTextureParameterName.TextureMinFilter, textureMinFilter);
    //    GL.TextureParameterI(texture, GlTextureParameterName.TextureMagFilter, textureMagFilter);

    //    return texture;
    //}

    //private static void ExtractMipmap(Bitmap bitmap, float[] pixels, int width, int height)
    //{
    //    int i = 0;
    //    for (int y = 0; y < height; y++)
    //    {
    //        for (int x = 0; x < width; x++)
    //        {
    //            var pixel = bitmap.GetPixel(x, y);
    //            pixels[i++] = pixel.R / 255f;
    //            pixels[i++] = pixel.G / 255f;
    //            pixels[i++] = pixel.B / 255f;
    //            pixels[i++] = pixel.A / 255f;
    //        }
    //    }
    //}

    //// could be improved using a Gauss-Filter
    //private static void GetMipmapAbove(float[] below, float[] above, int bWidth, int aWidth, int aHeight)
    //{
    //    int a = 0;
    //    for (int y = 0; y < aHeight; y++)
    //    {
    //        int b = 2 * y * 4 * bWidth;
    //        for (int x = 0; x < aWidth; x++)
    //        {
    //            for (int i = 0; i < 4; i++)
    //            {
    //                above[a] = below[b] +
    //                           below[b + 4] +
    //                           below[b + 4 * bWidth] +
    //                           below[b + 4 * (bWidth + 1)];
    //                above[a] *= 1f / 4f;
    //                a++; // next channel
    //                b++;
    //            }

    //            b += 4; // skip pixel
    //        }
    //    }
    //}
}