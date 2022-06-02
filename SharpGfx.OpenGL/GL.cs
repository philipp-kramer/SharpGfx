using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[assembly: InternalsVisibleTo("Test")]
namespace SharpGfx.OpenGL
{
    internal static class GL
    {
        [DllImport(@"x64/OpenGL.dll", EntryPoint = "createGlfWindow", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void* CreateWindow(string title, int width, int height);

        [DllImport(@"OpenGL.dll", EntryPoint = "isWindowCloseRequested", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe bool IsWindowCloseRequested(void* glfWindow);

        [DllImport(@"OpenGL.dll", EntryPoint = "swapBuffers", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void SwapBuffers(void* glfWindow);

        [DllImport(@"OpenGL.dll", EntryPoint = "getEvents", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void GetEvents(uint* data);

        [DllImport(@"OpenGL.dll", EntryPoint = "getMouseInputs", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void GetMouseInputs(double* data);

        [DllImport(@"OpenGL.dll", EntryPoint = "terminateGlfw", CallingConvention = CallingConvention.StdCall)]
        internal static extern void Terminate();


        [DllImport(@"OpenGL.dll", EntryPoint = "viewport", CallingConvention = CallingConvention.StdCall)]
        internal static extern void Viewport(int x, int y, int width, int height);
        [DllImport(@"OpenGL.dll", EntryPoint = "clearColor", CallingConvention = CallingConvention.StdCall)]
        internal static extern void ClearColor(float r, float g, float b, float a);

        [DllImport(@"OpenGL.dll", EntryPoint = "clear", CallingConvention = CallingConvention.StdCall)]
        internal static extern void Clear(GlBufferBit bufferBit);



        [DllImport(@"OpenGL.dll", EntryPoint = "genVertexArray", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint GenVertexArray();

        [DllImport(@"OpenGL.dll", EntryPoint = "genBuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint GenBuffer();

        [DllImport(@"OpenGL.dll", EntryPoint = "bufferData", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void BufferData(GlBufferTarget target, long size, void* data);

        [DllImport(@"OpenGL.dll", EntryPoint = "getBufferParameter", CallingConvention = CallingConvention.StdCall)]
        internal static extern long GetBufferParameter(GlBufferTarget target, GlBufferParameterName name);

        [DllImport(@"OpenGL.dll", EntryPoint = "floatVertexAttribPointer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void FloatVertexAttribPointer(uint index, int size, int stride, int offset);

        [DllImport(@"OpenGL.dll", EntryPoint = "bindVertexArray", CallingConvention = CallingConvention.StdCall)]
        internal static extern void BindVertexArray(uint array);

        [DllImport(@"OpenGL.dll", EntryPoint = "bindBuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void BindBuffer(GlBufferTarget target, uint buffer);

        [DllImport(@"OpenGL.dll", EntryPoint = "enableVertexAttribArray", CallingConvention = CallingConvention.StdCall)]
        internal static extern void EnableVertexAttribArray(uint array);

        [DllImport(@"OpenGL.dll", EntryPoint = "drawTriangles", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DrawTriangles(int first, int count);

        [DllImport(@"OpenGL.dll", EntryPoint = "drawIndexedTriangles", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void DrawIndexedTriangles(int count, GlType type, void* indices);

        [DllImport(@"OpenGL.dll", EntryPoint = "enable", CallingConvention = CallingConvention.StdCall)]
        internal static extern void Enable(GlCap cap);

        [DllImport(@"OpenGL.dll", EntryPoint = "disable", CallingConvention = CallingConvention.StdCall)]
        internal static extern void Disable(GlCap cap);

        [DllImport(@"OpenGL.dll", EntryPoint = "blendFunc", CallingConvention = CallingConvention.StdCall)]
        internal static extern void BlendFunc(GlBlendingFactor srcFactor, GlBlendingFactor dstFactor);


        [DllImport(@"OpenGL.dll", EntryPoint = "genTexture", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint GenTexture();

        [DllImport(@"OpenGL.dll", EntryPoint = "texImage2D", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void TexImage2D(GlTextureTarget target, int level, int width, int height, int border, GlPixelFormat format, GlPixelType type, void* pixels);

        [DllImport(@"OpenGL.dll", EntryPoint = "createTexture", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint CreateTexture(GlTextureTarget target);

        [DllImport(@"OpenGL.dll", EntryPoint = "textureStorage2D", CallingConvention = CallingConvention.StdCall)]
        internal static extern void TextureStorage2D(uint texture, int levels, GlSizedInternalFormat format, int width, int height);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "textureSubImage2D", CallingConvention = CallingConvention.StdCall)]
        internal static extern unsafe void TextureSubImage2D(uint texture, int level, int xoffset, int yoffset, int width, int height, GlPixelFormat format, GlPixelType type, void* pixels);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "textureParameterI", CallingConvention = CallingConvention.StdCall)]
        internal static extern void TextureParameterI(uint texture, GlTextureParameterName pname, GlTextureParameter param);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "activeTexture", CallingConvention = CallingConvention.StdCall)]
        internal static extern void ActiveTexture(GlTextureUnit glTexture);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "bindTexture", CallingConvention = CallingConvention.StdCall)]
        internal static extern void BindTexture(GlTextureTarget target, uint texture);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "deleteTexture", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DeleteTexture(uint texture);


        [DllImport(@"OpenGL.dll", EntryPoint = "genFramebuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint GenFramebuffer();

        [DllImport(@"OpenGL.dll", EntryPoint = "bindFramebuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void BindFramebuffer(GlFramebufferTarget target, uint framebuffer);

        [DllImport(@"OpenGL.dll", EntryPoint = "deleteFramebuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DeleteFramebuffer(uint framebuffer);

        [DllImport(@"OpenGL.dll", EntryPoint = "framebufferTexture2D", CallingConvention = CallingConvention.StdCall)]
        internal static extern void FramebufferTexture2D(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlTextureTarget textarget, uint texture, int level);

        [DllImport(@"OpenGL.dll", EntryPoint = "genRenderbuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint GenRenderbuffer();

        [DllImport(@"OpenGL.dll", EntryPoint = "bindRenderbuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void BindRenderbuffer(GlRenderbufferTarget target, uint renderbuffer);

        [DllImport(@"OpenGL.dll", EntryPoint = "renderbufferStorage", CallingConvention = CallingConvention.StdCall)]
        internal static extern void RenderbufferStorage(GlRenderbufferTarget target, GlRenderbufferStorage internalformat, int width, int height);

        [DllImport(@"OpenGL.dll", EntryPoint = "framebufferRenderbuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void FramebufferRenderbuffer(GlFramebufferTarget target, GlFramebufferAttachment attachment, GlRenderbufferTarget renderbuffertarget, uint renderbuffer);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "checkFramebufferStatus", CallingConvention = CallingConvention.StdCall)]
        internal static extern GlFramebufferErrorCode CheckFramebufferStatus(GlFramebufferTarget target);
        
        [DllImport(@"OpenGL.dll", EntryPoint = "deleteRenderbuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DeleteRenderbuffer(uint renderbuffer);

        [DllImport(@"OpenGL.dll", EntryPoint = "drawBuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DrawBuffer(uint buf);

        [DllImport(@"OpenGL.dll", EntryPoint = "readBuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void ReadBuffer(uint src);



        [DllImport(@"OpenGL.dll", EntryPoint = "compile", CallingConvention = CallingConvention.StdCall)]
        internal static extern uint Compile(string vertexShaderSource, string fragmentShaderSource);

        [DllImport(@"OpenGL.dll", EntryPoint = "useProgram", CallingConvention = CallingConvention.StdCall)]
        internal static extern void UseProgram(uint shaderProgram);

        [DllImport(@"OpenGL.dll", EntryPoint = "deleteProgram", CallingConvention = CallingConvention.StdCall)]
        public static extern void DeleteProgram(uint handle);


        [DllImport(@"OpenGL.dll", EntryPoint = "deleteVertexArray", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DeleteVertexArray(uint array);

        [DllImport(@"OpenGL.dll", EntryPoint = "deleteBuffer", CallingConvention = CallingConvention.StdCall)]
        internal static extern void DeleteBuffer(uint buffer);

        [DllImport(@"OpenGL.dll", EntryPoint = "getAttributeLocation", CallingConvention = CallingConvention.StdCall)]
        public static extern uint GetAttributeLocation(uint buffer, string attributeParameter);

        [DllImport(@"OpenGL.dll", EntryPoint = "getUniformLocation", CallingConvention = CallingConvention.StdCall)]
        public static extern uint GetUniformLocation(uint buffer, string name);


        [DllImport(@"OpenGL.dll", EntryPoint = "uniform1i", CallingConvention = CallingConvention.StdCall)]
        public static extern uint Uniform1(uint location, int value);

        [DllImport(@"OpenGL.dll", EntryPoint = "uniform1f", CallingConvention = CallingConvention.StdCall)]
        public static extern uint Uniform1(uint location, float value);

        [DllImport(@"OpenGL.dll", EntryPoint = "uniform2f", CallingConvention = CallingConvention.StdCall)]
        public static extern uint Uniform2(uint location, float v1, float v2);

        [DllImport(@"OpenGL.dll", EntryPoint = "uniform3f", CallingConvention = CallingConvention.StdCall)]
        public static extern uint Uniform3(uint location, float v1, float v2, float v3);

        [DllImport(@"OpenGL.dll", EntryPoint = "uniform4f", CallingConvention = CallingConvention.StdCall)]
        public static extern uint Uniform4(uint location, float v1, float v2, float v3, float v4);

        [DllImport(@"OpenGL.dll", EntryPoint = "uniformMatrix4f", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe uint UniformMatrix4(uint location, bool transpose, float* values);


        [DllImport(@"OpenGL.dll", EntryPoint = "getTexImage", CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe void GetTexImage(GlTextureTarget target, int level, GlPixelFormat format, GlPixelType type, void* pixels);
    }
}
