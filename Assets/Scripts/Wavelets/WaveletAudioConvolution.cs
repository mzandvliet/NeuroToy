using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

/*
Todo: 

- Fix high frequency garbage

- Allow runtime config of selected segment, resolution settings, recalculate
- Blue-noise jitter windows in sub-pixel tests
- Automatically respect Nyquist conditions

*/

public class WaveletAudioConvolution : MonoBehaviour
{
    [SerializeField] private AudioClip _clip;
    [SerializeField] private bool _savePng;

    private NativeArray<float> _audio;
    private NativeArray<float> _scaleogram;

    private Texture2D _scaleogramTex;

    const int _numPixPerScale = 4096/2;
    const int _numScales = 512;
    // const float _scalePowBase = 1.071f; // for 128
    const float _scalePowBase = 1.01f; // for 1024

    private void Awake() {
        int sr = _clip.frequency;

        _audio = new NativeArray<float>(sr, Allocator.Persistent); // _clip.samples

        var data = new float[_clip.samples];
        _clip.GetData(data, 0);
        for (int i = 0; i < _audio.Length; i++) {
            _audio[i] = data[sr * 3 + i];
        }

        _scaleogram = new NativeArray<float>(_numPixPerScale, Allocator.Persistent);

        _scaleogramTex = new Texture2D(_numPixPerScale, _numScales, TextureFormat.RFloat, 0, true);
        var tex = _scaleogramTex.GetPixelData<float>(0);

        var watch = System.Diagnostics.Stopwatch.StartNew();

        for (int scale = 0; scale < _numScales; scale++) {
            float freq = 20f + Mathf.Pow(1.017f, scale); // power law
            // float freq = math.lerp(1f, sr * 0.5f, scale / (float)_numScales); // linear
            Debug.LogFormat("Scale {0}, freq {1:0.00}", scale, freq);

            var trsJob = new TransformJob() {
                // in
                signal = _audio,
                freq = freq,
                sr = sr,

                // out
                scaleogram = _scaleogram,
            };
            trsJob.Schedule(_scaleogram.Length, 8, new JobHandle()).Complete();

            for (int x = 0; x < _numPixPerScale; x++) {
                tex[scale * _numPixPerScale + x] = _scaleogram[x];
            }
        }

        watch.Stop();
        Debug.LogFormat("Total time: {0} ms", watch.ElapsedMilliseconds);

        Normalize(tex);

        _scaleogramTex.Apply(false);

        if (_savePng) {
            ExportPNG();
        }
    }

    private void OnDestroy() {
        _audio.Dispose();
        _scaleogram.Dispose();
    }

    private float _guiX = 0;
    private float _guiY = 0;
    private float _guiXScale = 1;
    private float _guiYScale = 1;

    private void OnGUI() {
        GUI.DrawTexture(
            new Rect(_guiX, _guiY, _scaleogramTex.width * _guiXScale, _scaleogramTex.height * _guiYScale),
            _scaleogramTex
        );

        GUILayout.BeginVertical(GUILayout.Width(1000f));
        {
            _guiX = GUILayout.HorizontalSlider(_guiX, -Screen.width, -Screen.width);
            _guiY = GUILayout.HorizontalSlider(_guiY, -Screen.height, Screen.height);
            _guiXScale = GUILayout.HorizontalSlider(_guiXScale, 0.1f, 10f);
            _guiYScale = GUILayout.HorizontalSlider(_guiYScale, 0.1f, 10f);
        };
        GUILayout.EndVertical();
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

    [BurstCompile]
    public struct TransformJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> signal;
        [ReadOnly] public float freq;
        [ReadOnly] public int sr;

        [WriteOnly] public NativeArray<float> scaleogram;
        
        public void Execute(int p) {
            /*
            Todo: Calculate exact window size in samples needed to convolve current wavelet
            */

            Rng rng = new Rng((uint)(p * 123 + freq * 137));

            const int n = 3;
            int smpPerPix = signal.Length / _numPixPerScale;
            int smpPerPeriod = (int)math.floor(sr / freq) + 1;
            int smpPerWave = smpPerPeriod * n * 2;
            int windowStride = smpPerPeriod * 2;

            float dotSum = 0f;

            for (int i = 0; i < smpPerPix; i+=windowStride) {
                var windowJitter = 0;//rng.NextInt(smpPerPeriod >> 2);

                float waveDot = 0f;
                for (int w = 0; w < smpPerWave && p * smpPerPix + (i+1) * smpPerWave + windowJitter < signal.Length; w++) {
                    float waveTime = -n + (w / (float)smpPerWave) * (2f * n);
                    waveDot += Wave(waveTime, freq) * signal[p * smpPerPix + i * smpPerWave + windowJitter + w];
                }

                dotSum += math.abs(waveDot);
            }

            scaleogram[p] = dotSum;
        }
    }

    private static void Normalize(NativeArray<float> signal) {
        float max = 0f;
        for (int i = 0; i < signal.Length; i++)
        {
            signal[i] = math.log10(1f + signal[i]);

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
        const float n = 6; // todo: affects needed window size
        float s = n / (twopi * freq);
        return math.cos(twopi * time * freq) * math.exp(-(time*time) / (2f * s * s));
    }

    private void ExportPNG() {
        var pngPath = System.IO.Path.Combine(Application.dataPath, string.Format("{0}.png", System.DateTime.Now.ToFileTimeUtc()));
        var pngBytes = _scaleogramTex.EncodeToPNG();
        System.IO.File.WriteAllBytes(pngPath, pngBytes);
        Debug.LogFormat("Wrote image: {0}", pngPath);
    }
}
