using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;
using System.Runtime.CompilerServices;

public struct TransformConfig {
    public float convsPerPixMultiplier;

    public int texWidth;
    public int numScales;
    public float lowestScale;
    public float highestScale;
    public float scalePowBase;

    public float cyclesPerWave;

    public float _scaleNormalizationFactor;

    public void UpdateDerivedProperties() {
        /*
        Todo: this is cute, but note the divide-by-zero when scalePowBase == 1.0
        */
        _scaleNormalizationFactor = (1f / (Mathf.Pow(scalePowBase, numScales) - 1f)) * (highestScale - lowestScale);
    }
}

public static class WUtils {
    /*
    Wavelet design:

    https://www.wolframalpha.com/input/?i=cos(pi*2+*+t+*+f)+*+exp(-(t*t))+for+f+%3D+6%2C+t+%3D+-4+to+4
    https://www.wolframalpha.com/input/?i=plot+cos(pi*2+*+t+*+f)+*+exp(-(t^2)+%2F+(2+*+s^2))%2C+n+%3D+6%2C+f+%3D+10%2C+s+%3D+n+%2F+(pi*2*f)%2C++for+t+%3D+-6+to+6
    https://www.geogebra.org/calculator/wgetejw6

    */

    const float tau = math.PI * 2f;

    public static float Scale2Freq(float scale, TransformConfig cfg) {
        // power law
        return cfg.lowestScale + (Mathf.Pow(cfg.scalePowBase, scale) - 1f) * cfg._scaleNormalizationFactor;
    }

    public static float Freq2Scale(float freq, TransformConfig cfg) {
        // BUG: this doesn't work yet
        float a = freq / cfg._scaleNormalizationFactor;
        float b = a - cfg.lowestScale;
        float c = b + 1f;
        float d = Mathf.Pow(cfg.scalePowBase, 1f / b);
        return d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MorletReal(float time, float freq, float cyclesPerWave) {
        float s = WaveStdev(freq, cyclesPerWave);
        float phase = tau * time * freq;
        float gaussian = GaussianEnvelope(time, s);
        return math.cos(phase) * gaussian;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float2 MorletComplex(float time, float freq, float cyclesPerWave) {
        float s = WaveStdev(freq, cyclesPerWave);
        return GetComplexWave(time, freq) * GaussianEnvelope(time, s);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float2 GetComplexWave(float time, float freq) {
        float phase = tau * time * freq;
        float2 result;
        math.sincos(phase, out result.y, out result.x);
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveStdev(float freq, float cyclesPerWave) {
        return cyclesPerWave / (tau * freq);
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