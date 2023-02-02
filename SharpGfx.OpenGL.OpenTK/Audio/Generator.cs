using System;
using System.Collections.Generic;

namespace SharpGfx.OpenGL.OpenTK.Audio;

public static class Generator
{
    public readonly struct Tone
    {
        public readonly float Amplitude;
        public readonly int Frequency;

        public Tone(float amplitude, int frequency)
        {
            Amplitude = amplitude;
            Frequency = frequency;
        }
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    public static short[] GetSinSamples(List<Tone> tones)
    {
        const float dAngle = 2 * MathF.PI / Settings.SamplingFrequency;

        int gcd = tones[0].Frequency;
        for (int i = 1; i < tones.Count; i++)
        {
            gcd = Gcd(gcd, tones[i].Frequency);
        }
        int sampleCount = Settings.SamplingFrequency / gcd;

        var samples = new short[sampleCount * 2];
        for (int i = 0; i < samples.Length; ++i)
        {
            float angle = i * dAngle;
            float value = 0;
            foreach (var tone in tones)
            {
                value += tone.Amplitude * MathF.Sin(angle * tone.Frequency);
            }
            samples[i] = (short) (short.MaxValue * value / tones.Count);
        }

        return samples;
    }
}