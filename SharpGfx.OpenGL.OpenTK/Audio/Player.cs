//using System;
//using OpenTK;
//using OpenTK.Audio.OpenAL;

//namespace SharpGfx.OpenTK.Audio
//{
//    public sealed class Player : IDisposable
//    {
//        private readonly ContextHandle _context;
//        private readonly IntPtr _device;
//        private readonly int _source;

//        public Player(string deviceName)
//        {
//            _device = Alc.OpenDevice(deviceName);
//            unsafe
//            {
//                _context = Alc.CreateContext(_device, (int*)null);
//            }

//            Alc.MakeContextCurrent(_context);

//            AL.GenSources(1, out _source);
//        }

//        public void Play(short[] samples)
//        {
//            AL.GenBuffers(1, out int buffers);
//            AL.BufferData(buffers, ALFormat.Mono16, samples, samples.Length, Settings.SamplingFrequency);
//            AL.Source(_source, ALSourcei.Buffer, buffers);
//            AL.Source(_source, ALSourceb.Looping, true);

//            AL.SourcePlay(_source);
//        }

//        public void Stop()
//        {
//            AL.SourceStop(_source);
//        }

//        private void Dispose(bool disposing)
//        {
//            ReleaseUnmanagedResources();
//        }

//        private void ReleaseUnmanagedResources()
//        {
//            Alc.MakeContextCurrent(ContextHandle.Zero);
//            Alc.DestroyContext(_context);

//            Alc.CloseDevice(_device);
//        }

//        public void Dispose()
//        {
//            GC.SuppressFinalize(this);
//            Dispose(true);
//        }

//        ~Player()
//        {
//            Dispose(false);
//        }
//    }
//}