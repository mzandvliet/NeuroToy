using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

public static class WaveletUtils {
    /*
    Wavelet design:

    https://www.wolframalpha.com/input/?i=cos(pi*2+*+t+*+f)+*+exp(-(t*t))+for+f+%3D+6%2C+t+%3D+-4+to+4
    https://www.wolframalpha.com/input/?i=plot+cos(pi*2+*+t+*+f)+*+exp(-(t^2)+%2F+(2+*+s^2))%2C+n+%3D+6%2C+f+%3D+10%2C+s+%3D+n+%2F+(pi*2*f)%2C++for+t+%3D+-6+to+6
    https://www.geogebra.org/calculator/wgetejw6


    */

    public static float WaveReal(float time, float freq) {
        const float twopi = math.PI * 2f;
        const float n = 6; // todo: affects needed window size
        float s = n / (twopi * freq);

        float phase = twopi * time * freq;
        float gaussian = math.exp(-(time * time) / (2f * s * s));
        return math.cos(phase) * gaussian;
    }

    public static float2 WaveComplex(float time, float freq, float cyclesPerWave) {
        const float twopi = math.PI * 2f;
        float s = cyclesPerWave / (twopi * freq);

        float phase = twopi * time * freq;
        float gaussian = math.exp(-(time * time) / (2f * s * s));
        return new float2(
            math.cos(phase) * gaussian,
            math.sin(phase) * gaussian
        );
    }

    public static void ExportPNG(Texture2D tex) {
        var pngPath = System.IO.Path.Combine(Application.dataPath, string.Format("{0}.png", System.DateTime.Now.ToFileTimeUtc()));
        var pngBytes = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes(pngPath, pngBytes);
        Debug.LogFormat("Wrote image: {0}", pngPath);
    }
}