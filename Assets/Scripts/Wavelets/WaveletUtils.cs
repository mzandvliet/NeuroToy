using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;
using System.Runtime.CompilerServices;

public static class WUtils {
    /*
    Wavelet design:

    https://www.wolframalpha.com/input/?i=cos(pi*2+*+t+*+f)+*+exp(-(t*t))+for+f+%3D+6%2C+t+%3D+-4+to+4
    https://www.wolframalpha.com/input/?i=plot+cos(pi*2+*+t+*+f)+*+exp(-(t^2)+%2F+(2+*+s^2))%2C+n+%3D+6%2C+f+%3D+10%2C+s+%3D+n+%2F+(pi*2*f)%2C++for+t+%3D+-6+to+6
    https://www.geogebra.org/calculator/wgetejw6

    */

    const float twopi = math.PI * 2f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveReal(float time, float freq) {
        const float n = 6; // todo: affects needed window size
        float s = n / (twopi * freq);

        float phase = twopi * time * freq;
        float gaussian = GaussianEnvelope(time, s);
        return math.cos(phase) * gaussian;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float2 WaveComplex(float time, float freq, float cyclesPerWave) {
        float s = WaveStdev(time, freq, cyclesPerWave);
        float phase = twopi * time * freq;
        float gaussian = GaussianEnvelope(time, s);
        return new float2(
            math.cos(phase) * gaussian,
            math.sin(phase) * gaussian
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float2 GetWaveOsc(float timeStep, float freq, float cyclesPerWave) {
        float phase = twopi * timeStep * freq;
        return new float2(
            math.cos(phase),
            math.sin(phase)
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveStdev(float timeStep, float freq, float cyclesPerWave) {
        return cyclesPerWave / (twopi * freq);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GaussianEnvelope(float time, float stdev) {
        return math.exp(-(time * time) / (2f * stdev * stdev));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 CMul(float2 a, float2 b) {
        return new float2(
            a.x * b.x - a.y * b.y,
            a.x * b.y + a.y * b.x);
    }

    public static void ExportPNG(Texture2D tex) {
        var pngPath = System.IO.Path.Combine(Application.dataPath, string.Format("{0}.png", System.DateTime.Now.ToFileTimeUtc()));
        var pngBytes = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes(pngPath, pngBytes);
        Debug.LogFormat("Wrote image: {0}", pngPath);
    }
}