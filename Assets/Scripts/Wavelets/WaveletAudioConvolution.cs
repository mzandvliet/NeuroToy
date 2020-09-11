﻿using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

/*
Todo: 

- Add GUI control for changing scale layout
- Fix high frequency garbage
- Do rendering part on gpu, for live tweaking of scan results (contrast, etc)

- Automatically respect Nyquist conditions

Ideas:

- Monte carlo integration
    - Blue-noise jitter windows in sub-pixel tests
    - jitter in time, but also in scale/freq

- Quadtree datastructure for tracking measurements

It's funny, but in thinking of stochastically, adaptively sampling
the full continuous transform, the image we are building up can
itself be thought of as a wavelet-like structure

*/

public class WaveletAudioConvolution : MonoBehaviour
{
    [SerializeField] private AudioClip _clip;
    [SerializeField] private bool _savePng;

    private NativeArray<float> _audio;
    private NativeArray<float> _scaleogram;

    private Texture2D _scaleogramTex;

    private TransformConfig _config;
    private float _signalStart;
    private float _signalEnd;

    private void Awake() {
        _config = new TransformConfig()
        {
            numPixPerScale = 1024,
            numScales = 1024,
            scalePowBase = 1.009f, // for 1024

            waveTimeJitter = 0.0003f,
            waveFreqJitter = 0.0003f,
            convsPerPixMultiplier = 0.25f,
        };


        int sr = _clip.frequency;

        _audio = new NativeArray<float>(_clip.samples, Allocator.Persistent); // _clip.samples

        var data = new float[_clip.samples];
        _clip.GetData(data, 0);
        for (int i = 0; i < _audio.Length; i++) {
            _audio[i] = data[i];
        }

        _signalStart = 0f;
        _signalEnd = _clip.length;

        _scaleogram = new NativeArray<float>(_config.numPixPerScale, Allocator.Persistent);
        _scaleogramTex = new Texture2D(_config.numPixPerScale, _config.numScales, TextureFormat.RGBAFloat, 4, true);

        
    }

    private void OnDestroy() {
        _audio.Dispose();
        _scaleogram.Dispose();
    }

    private float _guiX = 8;
    private float _guiY = 8;
    private float _guiXScale = 1;
    private float _guiYScale = 1;

    private void OnGUI() {
        GUI.DrawTexture(
            new Rect(_guiX, _guiY, _scaleogramTex.width * _guiXScale, _scaleogramTex.height * _guiYScale),
            _scaleogramTex
        );

        GUILayout.BeginVertical(GUI.skin.box, GUILayout.Width(1000f));
        {
            GUILayout.Label("Position XY, Scale XY");
            _guiX = GUILayout.HorizontalSlider(_guiX, -Screen.width, Screen.width);
            _guiY = GUILayout.HorizontalSlider(_guiY, -Screen.height, Screen.height);
            _guiXScale = GUILayout.HorizontalSlider(_guiXScale, 0.1f, 10f);
            _guiYScale = GUILayout.HorizontalSlider(_guiYScale, 0.1f, 10f);

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Signal Start: {0:0.00} seconds", _signalStart));
            _signalStart = Mathf.Clamp(GUILayout.HorizontalSlider(_signalStart, 0f, _clip.length), 0f, _signalEnd-0.1f);
            GUILayout.Label(string.Format("Signal End: {0:0.00} seconds", _signalEnd));
            _signalEnd = Mathf.Clamp(GUILayout.HorizontalSlider(_signalEnd, 0f, _clip.length), _signalStart+0.1f, _clip.length);

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Resolution: {0} x {1}", _config.numPixPerScale, _config.numScales));
            _config.numPixPerScale = (int)math.pow(2, Mathf.RoundToInt(GUILayout.HorizontalSlider(math.log2(_config.numPixPerScale), 4, 12)));
            _config.numScales = (int)math.pow(2, Mathf.RoundToInt(GUILayout.HorizontalSlider(math.log2(_config.numScales), 4, 12)));

            GUILayout.Label(string.Format("Convs Per Pixel Multiplier: {0:0.00}", _config.convsPerPixMultiplier));
            _config.convsPerPixMultiplier = GUILayout.HorizontalSlider(_config.convsPerPixMultiplier, 0f, 4f);

            GUILayout.Label(string.Format("Wave Time Jitter: {0:0.000000}", _config.waveTimeJitter));
            _config.waveTimeJitter = GUILayout.HorizontalSlider(_config.waveTimeJitter, 0f, 0.01f);
            GUILayout.Label(string.Format("Wave Freq Jitter: {0:0.000000}", _config.waveFreqJitter));
            _config.waveFreqJitter = GUILayout.HorizontalSlider(_config.waveFreqJitter, 0f, 0.01f);
            

            if (GUILayout.Button("Transform")) {
                Transform();
            }
        };
        GUILayout.EndVertical();
    }

    private void Transform() {
        if (_scaleogram != null) {
            _scaleogram.Dispose();
            _scaleogram = new NativeArray<float>(_config.numPixPerScale, Allocator.Persistent);
            _scaleogramTex.Resize(_config.numPixPerScale, _config.numScales);
        }

        int sr = _clip.frequency;
        var tex = _scaleogramTex.GetPixelData<float4>(0);

        var signalSlice = _audio.Slice(
            (int)(_clip.frequency * _signalStart),
            (int)(_clip.frequency * (_signalEnd-_signalStart)));

        Debug.Log(_clip.frequency * _signalStart + ", " + _clip.frequency * _signalEnd);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        var handle = new JobHandle();

        for (int scale = 0; scale < _config.numScales; scale++) {
            float freq = 16f + Mathf.Pow(_config.scalePowBase, scale); // power law
            // float freq = math.lerp(1f, 1000f, scale / (float)(_config.numScales-1)); // linear
            // Debug.LogFormat("Scale {0}, freq {1:0.00}", scale, freq);

            var trsJob = new TransformJob()
            {
                // in
                signal = signalSlice,
                freq = freq,
                sr = sr,
                cfg = _config,

                // out
                scaleogram = _scaleogram,
            };
            handle = trsJob.Schedule(_scaleogram.Length, 8, handle);

            var copyJob = new CopyRowJob()
            {
                row = _scaleogram,
                targetTex = tex,
                rowIdx = scale,
                cfg = _config
            };
            handle = copyJob.Schedule(_scaleogram.Length, 32, handle);
        }

        var normJob = new NormalizeJob()
        {
            tex = tex,
        };
        handle = normJob.Schedule(handle);

        var visJob = new VisualizeJob()
        {
            tex = tex,
        };
        handle = visJob.Schedule(tex.Length, 32, handle);

        handle.Complete();

        watch.Stop();
        Debug.LogFormat("Total time: {0} ms", watch.ElapsedMilliseconds);

        _scaleogramTex.Apply(true);

        if (_savePng) {
            ExportPNG();
        }
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

    public struct TransformConfig {
        public float waveTimeJitter;
        public float waveFreqJitter;
        public float convsPerPixMultiplier;

        public int numPixPerScale;
        public int numScales;
        public float scalePowBase;

}

    [BurstCompile]
    public struct TransformJob : IJobParallelFor {
        [ReadOnly] public NativeSlice<float> signal;
        [ReadOnly] public float freq;
        [ReadOnly] public int sr;
        [ReadOnly] public TransformConfig cfg;

        [WriteOnly] public NativeSlice<float> scaleogram;
        
        public void Execute(int p) {
            /*
            Todo:
            - Let's fix this mess! Here's how:

            Center a wave kernel at the pixel position
            Consider that a 1hz wavelet with 6-period support will likely extend very far outside
            the range of samples 'covered' by a screen pixel.
            This does *not* mean that we chop the convolution to only within the space of the pixel,
            no, we still have to perform the whole convolution in so far as we have signal to work with
            */

            Rng rng = new Rng(0x52EAAEBBu + (uint)p * 0x5A9CA13Bu + (uint)(freq*128f) * 0xE0EB6C25u);

            const int n = 3;
            int smpPerPix = signal.Length / cfg.numPixPerScale;
            int smpPerPeriod = (int)math.floor(sr / freq) + 1;
            int smpPerWave = smpPerPeriod * n * 2;

            int convsPerPix = 1 + (int)math.round((smpPerPix / (float)smpPerWave) * cfg.convsPerPixMultiplier);

            int waveJitterMag = (int)(smpPerPeriod * cfg.waveTimeJitter);
            float freqJitterMag = cfg.waveFreqJitter / freq * smpPerPeriod;

            int convStep = smpPerPix / convsPerPix;

            float dotSum = 0f;

            for (int c = 0; c < convsPerPix; c++) {
                var smpStart = p * smpPerPix + c * convStep - smpPerWave / 2;
                var waveJitter = rng.NextInt(0, waveJitterMag);
                var freqJitter = rng.NextFloat(-freqJitterMag, freqJitterMag);

                float waveDot = 0f;
                for (int w = 0; w < smpPerWave; w++) {
                    float waveTime = -n + (w / (float)smpPerWave) * (2f * n);
                    int signalIdx = smpStart + w + waveJitter;

                    if (signalIdx < 0 || signalIdx >= signal.Length) {
                        continue;
                    }

                    waveDot += Wave(waveTime, freq + freqJitter) * signal[signalIdx];
                }

                dotSum += math.abs(waveDot);
            }

            scaleogram[p] = dotSum;
        }
    }

    [BurstCompile]
    public struct CopyRowJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> row;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float4> targetTex;
        [ReadOnly] public int rowIdx;
        [ReadOnly] public TransformConfig cfg;

        public void Execute(int i) {
            targetTex[(rowIdx * cfg.numPixPerScale + i)] = new float4(row[i]);
        }
    }

    [BurstCompile]
    public struct NormalizeJob : IJob {
        public NativeArray<float4> tex;

        public void Execute() {
            float max = 0f;
            for (int i = 0; i < tex.Length; i++) {
                tex[i] = new float4(math.log10(1f + tex[i].x), 0f, 0f, 0f);

                float mag = math.abs(tex[i].x);
                if (mag > max) {
                    max = mag;
                }
            }

            float maxInv = 1f / max;
            for (int i = 0; i < tex.Length; i++) {
                tex[i] *= maxInv;
            }
        }
    }

    [BurstCompile]
    public struct VisualizeJob : IJobParallelFor {
        public NativeArray<float4> tex;
        
        public void Execute(int i) {
            // tex[i] = new float4(
            //     tex[i].x,
            //     0f,
            //     0f,
            //     1f);

            // Spread across RGB
            // tex[i] = new float4(
            //     math.lerp(1f, 0f, math.abs(tex[i].x * 3f - 3f)),
            //     math.lerp(1f, 0f, math.abs(tex[i].x * 3f - 2f)),
            //     math.lerp(1f, 0f, math.abs(tex[i].x * 3f - 1f)),
            //     1f);

            // tex[i] = new float4(
            //    math.clamp(tex[i].x * 3f - 2f, 0f, 1f),
            //    math.clamp(tex[i].x * 3f - 1f, 0f, 1f),
            //    math.clamp(tex[i].x * 3f - 0f, 0f, 1f),
            //    1f);

            tex[i] = (Vector4)Color.HSVToRGB(tex[i].x, 1f, math.pow(tex[i].x, 0.5f));
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
