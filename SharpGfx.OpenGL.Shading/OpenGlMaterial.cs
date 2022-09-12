using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GlslParser;
using GlslParser.Tree;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public abstract class OpenGlMaterial : Material, IDisposable
    {
        private readonly Dictionary<string, Direction> _channels;
        private readonly Dictionary<string, Direction> _inputs;
        private readonly uint _handle;

        protected readonly Device Device;
        protected bool CheckUndefinedChannels { get; set; } = true;
        public bool Transparent { get; set; }

        protected OpenGlMaterial(Device device, string vertexShader, string fragShader)
        {
            Device = device;
            _channels = new Dictionary<string, Direction>();
            _inputs = new Dictionary<string, Direction>();

            var errors = new List<string>();

            CheckShader(vertexShader, errors);
            var outputChannels = CheckShader(fragShader, errors);
            if (outputChannels.Count != 1)
            {
                throw new ArgumentException($"the fragment shader must have a single output defining the color, but it has {outputChannels.Count}.");
            }

            Report(errors);

            _handle = device.Compile(vertexShader, fragShader, outputChannels.Single().Name, new List<string>());
        }

        protected override void DoInContext(Action action)
        {
            try
            {
                Device.UseProgram(_handle);
                action();
            }
            finally
            {
                Device.UseProgram(0);
            }
        }

        public override void Apply()
        {
            Device.UseProgram(_handle);
            if (Transparent) Device.EnableBlend();
        }

        public override void UnApply()
        {
            if (Transparent) Device.DisableBlend();
            Device.UseProgram(0);
        }

        public void SetVertexArrayAttributes(uint vertexArray, VertexAttribute[] attributes, VertexBuffer[] vertexBuffers)
        {
            DoInContext(() =>
            {
                Device.SetVertexArrayAttributes(vertexArray, _handle, attributes, vertexBuffers);
            });

            foreach (var attribute in attributes)
            {
                SetInput(attribute.Channel);
            }
        }

        protected List<ChannelNode> CheckShader(string shader, List<string> errors)
        {
            var reader = new StringReader(shader);
            var parser = new Parser(new Lexer(reader, errors), errors);
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

        public static void SetIfDefined(Space space, IEnumerable<OpenGlMaterial> materials, string name, IVector3 vector)
        {
            if (!vector.In(space)) throw new ArgumentException($"needs to be in {space.Domain}-space", nameof(vector));

            foreach (var material in materials)
            {
                material.DoInContext(() =>
                {
                    if (material._channels.ContainsKey(name))
                    {
                        material.Set(name, vector);
                    }
                });
            }
        }

        public static void Set(IEnumerable<OpenGlMaterial> materials, string name, Matrix4 matrix)
        {
            foreach (var material in materials)
            {
                material.DoInContext(() => material.Set(name, matrix));
            }
        }

        protected void SetInput(string channel)
        {
            _inputs[channel] = Direction.In;
        }

        private void SetUniform(string channel)
        {
            if (CheckUndefinedChannels)
            {
                if (!_channels.ContainsKey(channel))
                {
                    // TODO: add info in which resource the shader is stored
                    throw new ArgumentException($"shader channel {channel} not found");
                }
                if (_channels[channel] != Direction.Uniform)
                {
                    throw new ArgumentException("shader channel is not uniform", nameof(channel));
                }
            }

            _inputs[channel] = Direction.Uniform;
        }

        public void Set(string name, int value)
        {
            SetUniform(name);
            Device.Uniform1(Device.GetUniformLocation(_handle, name), value);
        }

        public void Set(string name, int[] values)
        {
            SetUniform(name);
            Device.Uniform1(Device.GetUniformLocation(_handle, name), values.Length, values);
        }

        public void Set(string name, float value)
        {
            SetUniform(name);
            Device.Uniform1(Device.GetUniformLocation(_handle, name), value);
        }

        public void Set(string name, float[] values)
        {
            SetUniform(name);
            Device.Uniform1(Device.GetUniformLocation(_handle, name), values.Length, values);
        }

        public void Set(string name, IVector2 value)
        {
            SetUniform(name);
            Device.Uniform2(Device.GetUniformLocation(_handle, name), value.X, value.Y);
        }

        public void Set(string name, IVector3 value)
        {
            SetUniform(name);
            Device.Uniform3(Device.GetUniformLocation(_handle, name), value.X, value.Y, value.Z);
        }

        public void Set(string name, ICollection<IVector3> values)
        {
            SetUniform(name);
            Device.Uniform3(Device.GetUniformLocation(_handle, name), values);
        }

        public void Set(string name, Vector4 value)
        {
            SetUniform(name);
            Device.Uniform4(Device.GetUniformLocation(_handle, name), value.X, value.Y, value.Z, value.W);
        }

        public void Set(string name, Matrix4 value)
        {
            SetUniform(name);
            Device.UniformMatrix4(Device.GetUniformLocation(_handle, name), value);
        }

        public void Reset(string input)
        {
            if (CheckUndefinedChannels && !_channels.ContainsKey(input)) throw new ArgumentException("shader channel not found", nameof(input));
            _inputs.Remove(input);
        }

        public void CheckInputs()
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
                else if (CheckUndefinedChannels && channel.Value != Direction.Out)
                {
                    throw new InvalidOperationException($"{kind} channel {channel.Key} has no input");
                }
            }
        }

        protected static void Report(List<string> errors)
        {
            foreach (var error in errors)
            {
                if (!string.IsNullOrEmpty(error)) Console.WriteLine(error);
            }
        }

        private void ReleaseUnmanagedResources()
        {
            Device.DeleteProgram(_handle);
        }

        protected virtual void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        ~OpenGlMaterial()
        {
            Dispose(false);
        }
    }
}