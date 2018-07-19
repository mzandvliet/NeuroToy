using UnityEngine;
using Unity.Mathematics;

public static class Complex2f {
    public const float Tau = Mathf.PI * 2f;

    public static float2 Real(float r) {
        return new float2(r, 0f);
    }

    public static float2 Imaginary(float i) {
        return new float2(0f, i);
    }

    public static float2 Mul(float2 a, float2 b) {
        return new float2(
            a.x * b.x - a.y * b.y,
            a.x * b.y + a.y * b.x);
    }

    public static float2 GetRotor(float freq, int samplerate) {
        float phaseStep = (Tau * freq) / samplerate;

        return new float2(
            math.cos(phaseStep),
            math.sin(phaseStep));
    }
}