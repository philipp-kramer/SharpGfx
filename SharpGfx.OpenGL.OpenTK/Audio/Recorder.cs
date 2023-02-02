//using System;
//using System.Collections.Generic;
//using OpenTK.Audio;
//using OpenTK.Audio.OpenAL;

//namespace SharpGfx.OpenTK.Audio
//{
//    public sealed class Recorder : IDisposable
//    {
//        public int TapeLength { get; }
//        public short[] Recording { get; }

//        private readonly AudioCapture _capture;

//        public Recorder(string deviceName, int tapeLength)
//        {
//            TapeLength = tapeLength;
//            _capture = new AudioCapture(deviceName, Settings.SamplingFrequency, ALFormat.Mono16, TapeLength);
//            Recording = new short[TapeLength];
//            _capture.Start();
//        }

//        public static IList<string> Devices => AudioCapture.AvailableDevices;

//        public string Device => _capture.CurrentDevice;

//        public void Record()
//        {
//            _capture.ReadSamples(Recording, TapeLength);
//        }

//        private void Dispose(bool disposing)
//        {
//            if (disposing)
//            {
//                _capture.Dispose();
//            }
//        }

//        public void Dispose()
//        {
//            GC.SuppressFinalize(this);
//            Dispose(true);
//        }

//        ~Recorder()
//        {
//            Dispose(false);
//        }
//    }
//}
