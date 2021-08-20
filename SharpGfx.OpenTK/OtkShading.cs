using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GlslParser;
using GlslParser.Tree;
using OpenTK.Graphics.OpenGL;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    public sealed class OtkShading : IDisposable
    {
        private readonly Dictionary<string, Direction> _channels;
        private readonly Dictionary<string, Direction> _inputs;
        private readonly int _handle;

        public bool UndefinedChannels { get; set; } = true;

        internal OtkShading(string vertexShader, string fragmentShader)
        {
            _channels = new Dictionary<string, Direction>();
            _inputs = new Dictionary<string, Direction>();

            var pipeline = new List<int>();
            var diagnosis = new Diagnosis();

            diagnosis.Source = "vertex shader";
            CheckShader(vertexShader, diagnosis);
            pipeline.Add(GetCompiledShader(vertexShader, ShaderType.VertexShader));

            diagnosis.Source = "fragment shader";
            var outputChannels = CheckShader(fragmentShader, diagnosis);
            if (outputChannels.Count != 1)
            {
                diagnosis.ReportError("fragment shaders must have a single output defining the color");
            }
            pipeline.Add(GetCompiledShader(fragmentShader, ShaderType.FragmentShader));

            if (diagnosis.HasErrors)
            {
                Report(diagnosis.ToString());
            }

            _handle = GL.CreateProgram();
            GL.BindFragDataLocation(_handle, 0, outputChannels.Single().Name);
            LinkProgram(_handle, pipeline);
        }

        private List<ChannelNode> CheckShader(string shader, Diagnosis diagnosis)
        {
            var reader = new StringReader(shader);
            var parser = new Parser(new Lexer(reader, diagnosis), diagnosis);
            var program = parser.ParseProgram();

            var channels = program.Declarations
                .OfType<ChannelNode>()
                .ToList();
            foreach (var channel in channels)
            {
                string channelName = channel.Name;
                var direction = channel.Direction;
                if (direction == Direction.In && _channels.TryGetValue(channelName, out var prev))
                {
                    if (prev == Direction.Out)
                    {
                        _channels.Remove(channelName);
                    }
                    else
                    {
                        throw new ArgumentException("matching shader channels require 'out' in vertex and 'in' in fragment shader", channelName);
                    }
                }
                else
                {
                    var declaration = program.Declarations.SingleOrDefault(d => d.Name == channel.Variable.Type.Name);
                    if (declaration is StructNode @struct)
                    {
                        foreach (var member in @struct.Members)
                        {
                            _channels.Add($"{channelName}.{member.Name}", direction);
                        }
                    }
                    else
                    {
                        _channels.Add(channelName, direction);
                    }
                }
            }

            return channels
                .Where(ch => ch.Direction == Direction.Out)
                .ToList();
        }

        private static int GetCompiledShader(string source, ShaderType shaderType)
        {
            var shader = GL.CreateShader(shaderType);
            GL.ShaderSource(shader, source);
            GL.CompileShader(shader);
            Report(GL.GetShaderInfoLog(shader));
            return shader;
        }

        private static void LinkProgram(int handle, List<int> shaders)
        {
            foreach (var shader in shaders)
            {
                GL.AttachShader(handle, shader);
            }
            GL.LinkProgram(handle);
            Report(GL.GetProgramInfoLog(handle));
            foreach (var shader in shaders)
            {
                GL.DetachShader(handle, shader);
                GL.DeleteShader(shader);
            }
        }

        internal int GetAttributeHandle(string name)
        {
            _inputs[name] = Direction.In;
            return GL.GetAttribLocation(_handle, name);
        }

        public void DoInContext(Action action)
        {
            try
            {
                GL.UseProgram(_handle);
                action();
            }
            finally
            {
                GL.UseProgram(0);
            }
        }

        internal void CheckInputs()
        {
            foreach (var channel in _channels)
            {
                if (_inputs.TryGetValue(channel.Key, out var kind))
                {
                    if (kind != channel.Value)
                    {
                        throw new InvalidOperationException($"input {kind} {channel.Key} should be {channel.Value}");
                    }
                }
                else if (UndefinedChannels && channel.Value != Direction.Out)
                {
                    throw new InvalidOperationException($"{kind} channel {channel.Key} has no input");
                }
            }
        }

        public void Set(string name, int value)
        {
            CheckAndSet(name);
            GL.Uniform1(GL.GetUniformLocation(_handle, name), value);
        }

        public void ResetInt(string name)
        {
            Set(name, 0);
            CheckAndReset(name);
        }

        public void Set(string name, int[] values)
        {
            CheckAndSet(name);
            GL.Uniform1(GL.GetUniformLocation(_handle, name), values.Length, values);
        }

        public void Set(string name, float value)
        {
            CheckAndSet(name);
            GL.Uniform1(GL.GetUniformLocation(_handle, name), value);
        }

        public void ResetFloat(string name)
        {
            Set(name, 0f);
            CheckAndReset(name);
        }

        public void Set(string name, float[] values)
        {
            CheckAndSet(name);
            GL.Uniform1(GL.GetUniformLocation(_handle, name), values.Length, values);
        }

        public void Set(string name, Vector2 value)
        {
            CheckAndSet(name);
            Set(name, ((OtkVector2)value).Value);
        }

        public void ResetVector2(string name)
        {
            CheckAndReset(name);
            Set(name, global::OpenTK.Vector2.Zero);
        }

        private void Set(string name, global::OpenTK.Vector2 value)
        {
            GL.Uniform2(GL.GetUniformLocation(_handle, name), ref value);
        }

        public void Set(string name, Vector3 value)
        {
            CheckAndSet(name);
            Set(name, ((OtkVector3)value).Value);
        }

        public void ResetVector3(string name)
        {
            CheckAndReset(name);
            Set(name, global::OpenTK.Vector3.Zero);
        }

        private void Set(string name, global::OpenTK.Vector3 value)
        {
            GL.Uniform3(GL.GetUniformLocation(_handle, name), ref value);
        }

        public void Set(string name, ICollection<Vector3> values)
        {
            CheckAndSet(name);
            var floats = new float[values.Count * 3];
            int j = 0;
            foreach (var vector in values)
            {
                floats[j++] = vector.X;
                floats[j++] = vector.Y;
                floats[j++] = vector.Z;
            }
            GL.Uniform3(GL.GetUniformLocation(_handle, name), floats.Length, floats);
        }

        public void Set(string name, Vector4 value)
        {
            CheckAndSet(name);
            Set(name, ((OtkVector4)value).Value);
        }

        public void ResetVector4(string name)
        {
            CheckAndReset(name);
            Set(name, global::OpenTK.Vector4.Zero);
        }

        private void Set(string name, global::OpenTK.Vector4 value)
        {
            GL.Uniform4(GL.GetUniformLocation(_handle, name), ref value);
        }

        public void Set(string name, Matrix4 value)
        {
            CheckAndSet(name);
            Set(name, ((OtkMatrix4) value).Value);
        }

        public void ResetZeroMatrix4(string name)
        {
            CheckAndReset(name);
            Set(name, global::OpenTK.Matrix4.Zero);
        }

        public void ResetIdentityMatrix4(string name)
        {
            CheckAndReset(name);
            Set(name, global::OpenTK.Matrix4.Identity);
        }

        private void Set(string name, global::OpenTK.Matrix4 value)
        {
            GL.UniformMatrix4(GL.GetUniformLocation(_handle, name), true, ref value);
        }

        private void CheckAndSet(string input)
        {
            if (UndefinedChannels)
            {
                if (!_channels.ContainsKey(input))
                {
                    throw new ArgumentException($"shader channel {input} not found", nameof(input));
                }
                if (_channels[input] != Direction.Uniform)
                {
                    throw new ArgumentException("shader channel is not uniform", nameof(input));
                }
            }

            _inputs[input] = Direction.Uniform;
        }

        private void CheckAndReset(string input)
        {
            if (UndefinedChannels && !_channels.ContainsKey(input)) throw new ArgumentException("shader channel not found", nameof(input));
            _inputs.Remove(input);
        }

        private static void Report(string message)
        {
            if (message != string.Empty)
            {
                Console.WriteLine(message);
            }
        }

        private void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteProgram(_handle);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }
    }
}
