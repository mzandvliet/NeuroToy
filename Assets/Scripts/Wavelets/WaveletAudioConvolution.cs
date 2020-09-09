using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

public class WaveletAudioConvolution : MonoBehaviour
{
    [SerializeField] private AudioClip _clip;

    private NativeArray<float> _audio;
    private NativeArray<float> _scaleogram;

    private Texture2D _scaleogramTex;

    const int _numPixPerScale = 128;
    const int _numScales = 8;

    private void Awake() {
        int sr = _clip.frequency;

        _audio = new NativeArray<float>(sr, Allocator.Persistent);
        
        var data = new float[_clip.samples];
        _clip.GetData(data, 0);
        for (int i = 0; i < _audio.Length; i++)
        {
            _audio[i] = data[sr * 6 + i];
        }

        _scaleogram = new NativeArray<float>(_numPixPerScale, Allocator.Persistent);

        _scaleogramTex = new Texture2D(_numPixPerScale, _numScales, TextureFormat.RFloat, 0, true);
        var tex = _scaleogramTex.GetPixelData<float>(0);

        for (int scale = 0; scale < _numScales; scale++)
        {
            float freq = math.pow(3.2f, scale);
            Transform(_audio, _scaleogram, freq, sr);

            for (int x = 0; x < _numPixPerScale; x++) {
                tex[scale * _numPixPerScale + x] = _scaleogram[x];
            }
        }

        Normalize(tex);

        _scaleogramTex.Apply(false);
    }

    private void OnDestroy() {
        _audio.Dispose();
        _scaleogram.Dispose();
    }

    private void OnGUI() {
        GUI.DrawTexture(
            new Rect(0f, 0f, Screen.width, Screen.height),
            _scaleogramTex
        );
    }

    private void TestWaveSampling() {
        float freq = 440f;
        int sr = _clip.frequency;
        int smpPerPeriod = (int)math.floor(sr / freq) + 1;
        int smpPerWave = smpPerPeriod * 2;
        for (int w = 0; w < smpPerWave; w++) {
            float waveTime = -1f + (w / (float)smpPerWave) * 2f;
            Debug.LogFormat("t: {0} -> {1}", waveTime, Wave(waveTime, freq));
        }
    }

    private static void Transform(NativeArray<float> signal, NativeArray<float> scaleogram, float freq, int sr) {
        // Convolve

        int smpPerPix = signal.Length / _numPixPerScale;
        
        
        int smpPerPeriod = (int)math.floor(sr / freq) + 1;
        int smpPerWave = smpPerPeriod * 2;
        int windowStride = smpPerPeriod * 1;

        for (int p = 0; p < _numPixPerScale; p++)
        {
            float dotSum = 0f;

            for (int i = p * smpPerPix; i < (p + 1) * smpPerPix && i+smpPerWave < signal.Length; i+= windowStride) {
                float waveDot = 0f;
                for (int w = 0; w < smpPerWave; w++) {
                    float waveTime = -1f + (w / (float)smpPerWave) * 2f;
                    waveDot += Wave(waveTime, freq) * signal[i+w];
                }

                dotSum += math.abs(waveDot);
            }

            scaleogram[p] = math.log10(1f + dotSum);
        }
    }

    private static void Normalize(NativeArray<float> signal) {
        float max = 0f;
        for (int i = 0; i < signal.Length; i++)
        {
            float mag = math.abs(signal[i]);
            if (mag > max) {
                max = mag;
            }
        }

        float maxInv = 1f / max;
        for (int i = 0; i < signal.Length; i++) {
            signal[i] *= maxInv;
        }
    }

    /*
    Wavelet design:

    https://www.wolframalpha.com/input/?i=cos(pi*2+*+t+*+f)+*+exp(-(t*t))+for+f+%3D+6%2C+t+%3D+-4+to+4
    https://www.wolframalpha.com/input/?i=plot+cos(pi*2+*+t+*+f)+*+exp(-(t^2)+%2F+(2+*+s^2))%2C+n+%3D+6%2C+f+%3D+10%2C+s+%3D+n+%2F+(pi*2*f)%2C++for+t+%3D+-6+to+6
    https://www.geogebra.org/calculator/wgetejw6


    */

    private static float Wave(float time, float freq) {
        const float twopi = math.PI * 2f;
        const float n = 6;
        float s = n / (twopi * freq);
        return math.cos(twopi * time * freq) * math.exp(-(time*time) / (2f * s * s));
    }
}
