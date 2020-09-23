using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

/*
Todo: 

- complex waves, store complex results as intermediates, leave amplitude or phase extraction to renderer
- variable wavelet kernel cycle count n (low-n for low freqs, high-n for high freqs, or adaptive)
- audio resynthesis
- study intensively the relationships between orthogonality, overcompleness, etc.
- Iterative rendering
- fix a memory leak
- Fix high frequency precision issues
- Do rendering part on gpu, for live tweaking of scan results (contrast, etc)
- Quadtree structure that supports logarithmic zooming/scrolling (smoothly changing the log parameter on scale distribution)

- Automatically respect Nyquist conditions

Ideas:

- Monte carlo integration
    - Blue-noise jitter windows in sub-pixel tests
    - jitter in time, but also in scale/freq

- Quadtree datastructure for tracking measurements

It's funny, but in thinking of stochastically, adaptively sampling
the full continuous transform, the image we are building up can
itself be thought of as a wavelet-like structure


---


Normalization and log scaling need to be done in a global sense


*/

public class WaveletAudioConvolution : MonoBehaviour
{
    [SerializeField] private AudioClip _clip;
    [SerializeField] private Renderer _renderer;

    private NativeArray<float> _audio;
    private NativeArray<float> _scaleogram;

    private Texture2D _scaleogramTex;

    private TransformConfig _config;
    private float _signalStart;
    private float _signalEnd;

    private AudioSource _source;

    private void Awake() {
        int sr = _clip.frequency;

        _config = new TransformConfig()
        {
            numPixPerScale = 1024,
            numScales = 1024,
            lowestScale = 16f,
            highestScale = sr / 2f,
            scalePowBase = 1.009f, // Todo: provide auto-normalization to highestScale regardless of chosen base

            cyclesPerWave = 3,

            waveTimeJitter = 0.0003f,
            waveFreqJitter = 0.0003f,
            convsPerPixMultiplier = 0.25f,
        };
        _config.UpdateDerivedProperties();

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
        _renderer.sharedMaterial.SetTexture("_MainTex", _scaleogramTex);

        _source = gameObject.AddComponent<AudioSource>();
        _source.clip = _clip;
    }

    private void OnDestroy() {
        _audio.Dispose();
        _scaleogram.Dispose();
    }

    private void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            if (_source.isPlaying) {
                _source.Stop();
            } else {
                _source.Stop();
                _source.time = _signalStart;
                _source.Play();
            }
        }

        if (_source.isPlaying && _source.time >= _signalEnd) {
            _source.Stop();
        }

        float normalizedTime = (_source.time - _signalStart) / (_signalEnd - _signalStart);
        _renderer.sharedMaterial.SetFloat("_playTime", normalizedTime);
    }

    private float _guiX = 8;
    private float _guiY = 8;
    private float _guiXScale = 1;
    private float _guiYScale = 1;

    private void OnGUI() {
        // GUI.DrawTexture(
        //     new Rect(_guiX, _guiY, _scaleogramTex.width * _guiXScale, _scaleogramTex.height * _guiYScale),
        //     _scaleogramTex
        // );

        GUILayout.BeginVertical();
        {
            const int labelHeight = 32;
            float scaledHeight = _scaleogramTex.height * _guiYScale;
            int numScaleLabels = Mathf.RoundToInt(scaledHeight / labelHeight);
            float scaleStep = (float)_config.numScales / (numScaleLabels-1);
            float heightStep = scaledHeight / (numScaleLabels-1);
            for (int i = 0; i < numScaleLabels; i++) {
                GUI.Label(new Rect(
                    _guiX,
                    _guiY + scaledHeight - heightStep * i - labelHeight * 0.5f,
                    100f,
                    labelHeight),
                    string.Format("{0}Hz", Scale2Freq(i * scaleStep, _config)));
            }
        }
        GUILayout.EndVertical();

        GUILayout.BeginVertical(GUI.skin.box, GUILayout.Width(1000f));
        {
            GUILayout.Label("Position XY, Scale XY");
            _guiX = GUILayout.HorizontalSlider(_guiX, -Screen.width, Screen.width);
            _guiY = GUILayout.HorizontalSlider(_guiY, -Screen.height, Screen.height);
            _guiXScale = GUILayout.HorizontalSlider(_guiXScale, 0.1f, 10f);
            _guiYScale = GUILayout.HorizontalSlider(_guiYScale, 0.1f, 10f);

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Signal Start: {0:0.00} seconds", _signalStart));
            _signalStart = Mathf.Clamp(GUILayout.HorizontalSlider(_signalStart, 0f, _clip.length), 0f, _signalEnd-0.01f);
            GUILayout.Label(string.Format("Signal End: {0:0.00} seconds", _signalEnd));
            _signalEnd = Mathf.Clamp(GUILayout.HorizontalSlider(_signalEnd, 0f, _clip.length), _signalStart+0.01f, _clip.length);

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Lowest Scale {0:0.00} Hz", _config.lowestScale));
            _config.lowestScale = Mathf.Clamp(GUILayout.HorizontalSlider(_config.lowestScale, 0f, _clip.frequency/2f), 0f, _config.highestScale - 1f);
            GUILayout.Label(string.Format("Highest Scale {0:0.00} Hz", _config.highestScale));
            _config.highestScale = Mathf.Clamp(GUILayout.HorizontalSlider(_config.highestScale, _config.lowestScale, _clip.frequency/2f), _config.lowestScale + 1f, _clip.frequency / 2f);
            GUILayout.Label(string.Format("Scale Power Base {0:0.00}", _config.scalePowBase));
            _config.scalePowBase = GUILayout.HorizontalSlider(_config.scalePowBase, 1.000001f, 1.25f);

            _config.UpdateDerivedProperties();

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Resolution: {0} x {1}", _config.numPixPerScale, _config.numScales));
            _config.numPixPerScale = (int)math.pow(2, Mathf.RoundToInt(GUILayout.HorizontalSlider(math.log2(_config.numPixPerScale), 4, 12)));
            _config.numScales = (int)math.pow(2, Mathf.RoundToInt(GUILayout.HorizontalSlider(math.log2(_config.numScales), 4, 12)));

            GUILayout.Label(string.Format("Convs Per Pixel Multiplier: {0:0.00}", _config.convsPerPixMultiplier));
            _config.convsPerPixMultiplier = GUILayout.HorizontalSlider(_config.convsPerPixMultiplier, 0f, 64f);

            GUILayout.Label(string.Format("Wave Time Jitter: {0:0.000000}", _config.waveTimeJitter));
            _config.waveTimeJitter = GUILayout.HorizontalSlider(_config.waveTimeJitter, 0f, 0.01f);
            GUILayout.Label(string.Format("Wave Freq Jitter: {0:0.000000}", _config.waveFreqJitter));
            _config.waveFreqJitter = GUILayout.HorizontalSlider(_config.waveFreqJitter, 0f, 0.1f);
            
            GUILayout.BeginHorizontal();
            {
                if (GUILayout.Button("Transform")) {
                    Transform();
                }

                if (GUILayout.Button("Export PNG")) {
                    ExportPNG();
                }
            }
            GUILayout.EndHorizontal();
        }
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

        var watch = System.Diagnostics.Stopwatch.StartNew();
        var handle = new JobHandle();

        for (int scale = 0; scale < _config.numScales; scale++) {
            float freq = Scale2Freq(scale, _config);
            // Debug.LogFormat("Scale {0}, freq {1:0.00}", scale, freq);

            int smpPerPeriod = (int)math.ceil(sr / freq);
            // Debug.LogFormat("Scale {0}, freq {1}, smpPerPeriod {2}", scale, freq, smpPerPeriod);

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
    }

    private void TestWaveSampling() {
        float freq = 440f;
        int sr = _clip.frequency;
        int smpPerPeriod = (int)math.floor(sr / freq) + 1;
        int smpPerWave = smpPerPeriod * 2;
        for (int w = 0; w < smpPerWave; w++) {
            float waveTime = -1f + (w / (float)smpPerWave) * 2f;
            Debug.LogFormat("t: {0} -> {1}", waveTime, WaveReal(waveTime, freq));
        }
    }

    public struct TransformConfig {
        public float waveTimeJitter;
        public float waveFreqJitter;
        public float convsPerPixMultiplier;

        public int numPixPerScale;
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
            - Remaining high frequency issues
            */

            Rng rng = new Rng(0x52EAAEBBu + (uint)p * 0x5A9CA13Bu + (uint)(freq*0xCD0445A5u) * 0xE0EB6C25u);

            float nHalf = cfg.cyclesPerWave / 2f;
            float smpPerPix = signal.Length / (float)cfg.numPixPerScale;
            float smpPerPixHalf = (smpPerPix) * 0.5f;
            float smpPerPeriod = sr / freq;
            float smpPerWave = smpPerPeriod * nHalf * 2f;

            int convsPerPix = (int)math.ceil(((smpPerPix / smpPerWave) * cfg.convsPerPixMultiplier));

            int waveJitterMag = 1 + (int)(smpPerPeriod * cfg.waveTimeJitter);
            float freqJitterMag = cfg.waveFreqJitter / freq * smpPerPeriod;

            float convStep = smpPerPix / (float)convsPerPix;

            float timeSpan = 1f / freq * nHalf;

            float dotSum = 0f;

            for (int c = 0; c < convsPerPix; c++) {
                // float smpStart = p * smpPerPix + c * convStep;
                
                float smpStart = p * smpPerPix + 0.5f * smpPerPix - 0.5f * smpPerWave + rng.NextFloat(-smpPerPixHalf, smpPerPixHalf) * rng.NextFloat(0f, .5f);

                float waveDot = 0f;
                for (int w = 0; w <= smpPerWave; w++) {
                    float waveTime = -timeSpan + (w / smpPerWave) * (timeSpan * 2f);
                    int signalIdx = (int)(smpStart + w);

                    if (signalIdx < 0 || signalIdx >= signal.Length) {
                        continue;
                    }

                    float2 wave = WaveComplex(waveTime, freq, cfg);
                    wave = Mul(wave, new float2(signal[signalIdx], 0f));

                    waveDot += wave.x;
                }

                dotSum += math.abs(waveDot) / smpPerWave;
            }

            scaleogram[p] = dotSum / (float)convsPerPix;
        }

        public static Vector2 Mul(float2 a, float2 b) {
            return new float2(
                a.x * b.x - a.y * b.y,
                a.x * b.y + a.y * b.x);
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
                float mag = math.log10(1f + tex[i].x);
                // float mag = tex[i].x;
                tex[i] = new float4(mag, 0f, 0f, 0f);

                mag = math.abs(tex[i].x);
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

            tex[i] = (Vector4)Color.HSVToRGB(tex[i].x, 1f, math.pow(tex[i].x, 1f));
        }
    }

    /*
    Wavelet design:

    https://www.wolframalpha.com/input/?i=cos(pi*2+*+t+*+f)+*+exp(-(t*t))+for+f+%3D+6%2C+t+%3D+-4+to+4
    https://www.wolframalpha.com/input/?i=plot+cos(pi*2+*+t+*+f)+*+exp(-(t^2)+%2F+(2+*+s^2))%2C+n+%3D+6%2C+f+%3D+10%2C+s+%3D+n+%2F+(pi*2*f)%2C++for+t+%3D+-6+to+6
    https://www.geogebra.org/calculator/wgetejw6


    */

    private static float WaveReal(float time, float freq) {
        const float twopi = math.PI * 2f;
        const float n = 6; // todo: affects needed window size
        float s = n / (twopi * freq);

        float phase = twopi * time * freq;
        float gaussian = math.exp(-(time * time) / (2f * s * s));
        return math.cos(phase) * gaussian;
    }

    private static float2 WaveComplex(float time, float freq, TransformConfig cfg) {
        const float twopi = math.PI * 2f;
        float s = cfg.cyclesPerWave / (twopi * freq);

        float phase = twopi * time * freq;
        float gaussian = math.exp(-(time * time) / (2f * s * s));
        return new float2(
            math.cos(phase) * gaussian,
            math.sin(phase) * gaussian
        );
    }

    private static float Scale2Freq(float scale, TransformConfig cfg) {
        // linear
        // return math.lerp(1f, 1000f, scale / (float)(_config.numScales-1)); 

        // power law
        return cfg.lowestScale + (Mathf.Pow(cfg.scalePowBase, scale) - 1f) * cfg._scaleNormalizationFactor;
    }

    private void ExportPNG() {
        var pngPath = System.IO.Path.Combine(Application.dataPath, string.Format("{0}.png", System.DateTime.Now.ToFileTimeUtc()));
        var pngBytes = _scaleogramTex.EncodeToPNG();
        System.IO.File.WriteAllBytes(pngPath, pngBytes);
        Debug.LogFormat("Wrote image: {0}", pngPath);
    }
}
