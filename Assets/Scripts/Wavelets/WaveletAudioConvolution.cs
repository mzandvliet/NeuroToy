using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;
using Shapes;

/*
Todo: 

- Understand wavelet orthogonality
- store complex results as intermediates, leave amplitude or phase extraction to renderer
- variable wavelet kernel cycle count n (low-n for low freqs, high-n for high freqs, or adaptive)
- fit parameterized curves to trends and momenta (tracking frequency, phase, amplitude)
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

- Optimizations
    - Skip samples in convolution
        - What if you skip every other sample, or only multiply one in every p samples in a single wave convolution?
        - Maybe a skip-step that is relatively prime to the frequency of the wave?
    - Early-out in areas where preliminary tests reveal little to no relevant energy
    - Prefilter and lower samplerate before convolving lower-frequency parts of the signal
    - Convolve superpositions of many waves
        - Interleave harmonic wave convolutions, since they share samples.
        - Or interleave relatively prime waves since they do not.


It's funny, but in thinking of stochastically, adaptively sampling
the full continuous transform, the image we are building up can
itself be thought of as a wavelet-like structure


---


Normalization and log scaling need to be done in a global sense


*/

public struct TimeRange {
    public double start;
    public double duration;

    public TimeRange(double start, double duration) {
        this.start = start;
        this.duration = duration;
    }
}

public class WaveletAudioConvolution : MonoBehaviour
{
    [SerializeField] private AudioClip _clip;
    [SerializeField] private Renderer _renderer;
    [SerializeField] private Camera _camera;
    [SerializeField] private MeshRenderer _scaleogramRenderer;

    private NativeArray<float> _signal;

    private Texture2D _scaleogramTex;

    private TransformConfig _config;
    private float _signalStart;
    private float _signalEnd;
    private float _samplerate;

    private AudioSource _source;
    private NativeList<int> _primes;

    private void Awake() {
        Application.runInBackground = true;
        Application.targetFrameRate = 60;
        Camera.onPreRender += OnPreRenderCallback;

        _samplerate = _clip.frequency;

        _config = new TransformConfig()
        {
            texWidth = 1024,
            numScales = 1024,
            lowestScale = 16f,
            highestScale = _samplerate / 2f,
            scalePowBase = 1.009f,

            cyclesPerWave = 8f,

            convsPerPixMultiplier = 0.25f,
        };
        _config.UpdateDerivedProperties();

        _signal = new NativeArray<float>(_clip.samples, Allocator.Persistent); 

        var data = new float[_clip.samples];
        _clip.GetData(data, 0);
        for (int i = 0; i < _signal.Length; i++) {
            _signal[i] = data[i];
        }

        _signalStart = 0f;
        _signalEnd = _clip.length;

        _scaleogramTex = new Texture2D(_config.texWidth, _config.numScales, TextureFormat.RFloat, 0, true);
        _scaleogramTex.anisoLevel = 8;
        _scaleogramTex.wrapMode = TextureWrapMode.Clamp;
        _renderer.sharedMaterial.SetTexture("_MainTex", _scaleogramTex);

        _source = gameObject.AddComponent<AudioSource>();
        _source.clip = _clip;

        _primes = new NativeList<int>(Allocator.Persistent);
        _primes.Add(1); // LOL
        _primes.Add(2);
        _primes.Add(3);
        _primes.Add(5);
        _primes.Add(7);
        _primes.Add(11);
        _primes.Add(13);
        _primes.Add(17);
        _primes.Add(19);
        _primes.Add(23);
        _primes.Add(29);
        _primes.Add(31);
        _primes.Add(37);
        _primes.Add(41);
        _primes.Add(43);
        _primes.Add(47);
        _primes.Add(53);
        _primes.Add(59);
        _primes.Add(61);
        _primes.Add(67);
        _primes.Add(71);
        _primes.Add(73);
        _primes.Add(79);
        _primes.Add(83);
        _primes.Add(89);
        _primes.Add(97);
        _primes.Add(101);
        _primes.Add(103);
        _primes.Add(107);
        _primes.Add(109);
        _primes.Add(113);
        _primes.Add(127);
    }

    private void OnDestroy() {
        _signal.Dispose();
        _primes.Dispose();
    }

    private void Update() {
        if (_scaleogramTex.width != _config.texWidth || _scaleogramTex.height != _config.numScales) {
            _scaleogramTex.Resize(_config.texWidth, _config.numScales);
        }

        UpdatePlayback();
        UpdateCamera();
    }

    

    private void UpdatePlayback() {
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
        _renderer.sharedMaterial.SetFloat("_bias", _bias);
        _renderer.sharedMaterial.SetFloat("_gain", _gain);
    }

    private float3 _smoothScroll;
    private float2 _smoothZoom;
    private float2 _zoomLevel = new float2(1, 1);
    private void UpdateCamera() {
        float3 scrollInput = float3.zero;
        float2 zoomInput = float2.zero;

        if (Input.GetMouseButton(0)) {
            if (Input.GetKey(KeyCode.LeftShift)) {
                zoomInput = new float2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
            } else {
                scrollInput = new float3(-Input.GetAxis("Mouse X"), -Input.GetAxis("Mouse Y"), 0f);
            }
        }
        _smoothScroll = math.lerp(_smoothScroll, scrollInput, Time.deltaTime * 10f);
        _camera.transform.Translate(_smoothScroll);

        _smoothZoom = math.lerp(_smoothZoom, zoomInput, Time.deltaTime * 10f);
        _zoomLevel += _smoothZoom;
        _zoomLevel = math.clamp(_zoomLevel, new float2(0.1f, 0.1f), new float2(20f, 50f));
    }

    private void OnPreRenderCallback(Camera cam) {
        /*
        Todo: run all state transformations here as async burst jobs, then here we just push to Shapes
        */

        if (!_signal.IsCreated) {
            return;
        }

        var data = _signal;

        var drawRect = new Rect(0f, 0f, 240f * _zoomLevel.x, 10f * _zoomLevel.y);

        var timeRange = new TimeRange((long)_signalStart, _signalEnd - _signalStart);
       
        // DrawScaleLabels(cam, drawRect, _cfg);
        // DrawHorizontalLabels(cam, drawRect, _trades, timeRange);

        DrawControls(_camera, drawRect, _config, timeRange);

        var cameraLocalPos = _scaleogramRenderer.transform.InverseTransformPoint(_camera.transform.position);
        _scaleogramRenderer.transform.localScale = new Vector3(
            drawRect.width, drawRect.height, 1f
        );
        _scaleogramRenderer.transform.position = new Vector3(
            drawRect.width / 2, drawRect.height / 2, 0f
        );
        _camera.transform.position = _scaleogramRenderer.transform.TransformPoint(cameraLocalPos);
    }

    private void DrawControls(Camera cam, Rect area, TransformConfig cfg, TimeRange timeRange) {
        double startTime = timeRange.start;
        double endTime = (timeRange.start + timeRange.duration);
        double timeSpan = timeRange.duration;

        using (Draw.Command(cam)) {
            float nowDrawTime = (float)(((_source.time - startTime) / timeSpan) * area.width);
            float3 nowPos = new float3(
                new float2(area.x, area.y) + new float2(
                    nowDrawTime,
                    0
                ),
                0f
            );
            // Draw.Rectangle(nowPos, new Rect(0f, 0f, 1f * area.width, area.height), Color.blue);
            Draw.Line(nowPos, nowPos + new float3(0, area.height, 0), Color.white);
        }
    }

    private float _guiX = 8;
    private float _guiY = 8;
    private float _guiXScale = 1;
    private float _guiYScale = 1;
    private float _bias = 0f;
    private float _gain = 1f;

    private void OnGUI() {
        float drawSize = Mathf.Min(Screen.width, Screen.height);
        // GUI.DrawTexture(new Rect(0, 0, drawSize, drawSize), _scaleogramTex);

        GUILayout.BeginVertical(GUI.skin.box, GUILayout.Width(1000f));
        {
            GUILayout.Label("Position XY, Scale XY");
            _guiX = GUILayout.HorizontalSlider(_guiX, -Screen.width, Screen.width);
            _guiY = GUILayout.HorizontalSlider(_guiY, -Screen.height, Screen.height);
            _guiXScale = GUILayout.HorizontalSlider(_guiXScale, 0.1f, 10f);
            _guiYScale = GUILayout.HorizontalSlider(_guiYScale, 0.1f, 10f);

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Signal Start: {0:0.00} periods", _signalStart));
            _signalStart = Mathf.Clamp(GUILayout.HorizontalSlider(_signalStart, 0f, _signal.Length / _samplerate), 0f, _signalEnd - 0.01f);
            GUILayout.Label(string.Format("Signal End: {0:0.00} periods", _signalEnd));
            _signalEnd = Mathf.Clamp(GUILayout.HorizontalSlider(_signalEnd, 0f, _signal.Length / _samplerate), _signalStart + 0.01f, _signal.Length / _samplerate);
            GUILayout.Label(string.Format("Signal's max freq: {0:0.00} Hz", math.min(_samplerate / 2f, ((_signalEnd - _signalStart) * (_samplerate / 2f)))));
            GUILayout.Label(string.Format("Signal's min freq: {0:0.00} Hz", _config.cyclesPerWave / (_signalEnd - _signalStart)));

            GUILayout.Space(16f);
            GUILayout.Label(string.Format("Cycles per morlet wave {0}", _config.cyclesPerWave));
            _config.cyclesPerWave = Mathf.Clamp((int)math.round(GUILayout.HorizontalSlider(_config.cyclesPerWave, 1, 32)), 1, 32);
            GUILayout.Label(string.Format("Lowest Scale {0:0.00} cycles/period", _config.lowestScale));
            _config.lowestScale = -1f + math.pow(2f, Mathf.Clamp(GUILayout.HorizontalSlider(math.log2(1f + _config.lowestScale), math.log2(1f + 0.06f), math.log2(1f + _samplerate / 2f)), 0f, math.log2(1f + _config.highestScale - 1)));
            GUILayout.Label(string.Format("Highest Scale {0:0.00} cycles/period", _config.highestScale));
            _config.highestScale = Mathf.Clamp(GUILayout.HorizontalSlider(_config.highestScale, _config.lowestScale, _samplerate / 2f), _config.lowestScale + 1f, _samplerate / 2f);
            GUILayout.Label(string.Format("Scale Power Base {0:0.0000}", _config.scalePowBase));
            _config.scalePowBase = GUILayout.HorizontalSlider(_config.scalePowBase, 1.0f, 1.25f);

            _config.UpdateDerivedProperties();

            GUILayout.Space(16f);

            GUILayout.Label(string.Format("Resolution: {0} x {1}", _config.texWidth, _config.numScales));
            _config.texWidth = (int)math.pow(2, Mathf.RoundToInt(GUILayout.HorizontalSlider(math.log2(_config.texWidth), 4, 16)));
            _config.numScales = (int)math.pow(2, Mathf.RoundToInt(GUILayout.HorizontalSlider(math.log2(_config.numScales), 4, 12)));
            GUILayout.Label(string.Format("Tex size: {0}, bytes: {1}", _config.texWidth * _config.numScales, _config.texWidth * _config.numScales * 4));
            GUILayout.Label(string.Format("Tex max freq: {0:0.00} Hz", ((_signalEnd - _signalStart) * (_samplerate / 2f)) / _config.texWidth));

            GUILayout.Label(string.Format("Convs Per Pixel Multiplier: {0:0.00}", _config.convsPerPixMultiplier));
            _config.convsPerPixMultiplier = GUILayout.HorizontalSlider(_config.convsPerPixMultiplier, 0.001f, 32f);

            GUILayout.BeginHorizontal();
            {
                GUI.enabled = _transformRoutine == null;
                if (GUILayout.Button("Transform")) {
                    _transformRoutine = StartCoroutine(TransformAsync());
                }

                // if (GUILayout.Button("Export PNG")) {
                //     string pngPath = Path.Combine(Util.GetProjectFolder("Datasets"), $"{_dataName}.png");
                //     WUtils.SavePng(_scaleogramTex, pngPath);
                // }

                // if (GUILayout.Button("Export Floats")) {
                //     string path = Path.Combine(Util.GetProjectFolder("Datasets"), $"{_dataName}.floats");
                //     WUtils.SaveRawFloats(_scaleogramTex.GetPixelData<float>(0), path);
                // }
            }
            GUILayout.EndHorizontal();
        }
        GUILayout.EndVertical();
    }

    private Coroutine _transformRoutine;

    private System.Collections.IEnumerator TransformAsync() {
        var scaleogramLine = new NativeArray<float>(_config.texWidth, Allocator.Persistent);
        var tex = _scaleogramTex.GetPixelData<float>(0);

        var signalSlice = _signal.Slice(
            (int)(_samplerate * _signalStart),
            (int)(_samplerate * (_signalEnd - _signalStart)));

        var watch = System.Diagnostics.Stopwatch.StartNew();
        var handle = new JobHandle();

        for (int scale = 0; scale < _config.numScales; scale++) {
            float freq = WUtils.Scale2Freq(scale, _config);

            var trsJob = new TransformJob()
            {
                // in
                signal = signalSlice,
                centerFreq = freq,
                sr = (int)_samplerate,
                cfg = _config,
                tick = (uint)Time.frameCount,
                primes = _primes,

                // out
                scaleogram = scaleogramLine,
            };
            /* Todo: multiply batchCount by inverse of scale, such that
            the longer the wavelength, the more it is chopped up
            into smaller batches. Otherwise you end up with
            bubbles, and waiting workers. */
            handle = trsJob.Schedule(scaleogramLine.Length, 8, handle);

            var toTexJob = new CopyRowJob()
            {
                row = scaleogramLine,
                targetTex = tex,
                rowIdx = scale,
                cfg = _config
            };
            handle = toTexJob.Schedule(scaleogramLine.Length, 32, handle);
        }

        while (!handle.IsCompleted) {
            yield return new WaitForEndOfFrame();
        }
        handle.Complete();

        watch.Stop();
        Debug.LogFormat("Total time: {0} ms", watch.ElapsedMilliseconds);

        _scaleogramTex.Apply(true);

        scaleogramLine.Dispose();

        _transformRoutine = null;
    }

    [BurstCompile]
    public struct TransformJob : IJobParallelFor {
        [ReadOnly] public NativeSlice<float> signal;
        [ReadOnly] public float centerFreq;
        [ReadOnly] public int sr;
        [ReadOnly] public TransformConfig cfg;
        [ReadOnly] public uint tick;
        [ReadOnly] public NativeArray<int> primes;

        [WriteOnly] public NativeSlice<float> scaleogram;

        public void Execute(int p) {
            /*
            Todo:
            - Keep track of highest amplitude found for free normalization

            - Interpolated sampling for high frequency content, such that
            we no longer get artifacts due to neighbouring frequencies
            convolving from exactly the same discrete place in sample-space

            For lowest freqs, various forms of undersampling are fine. This
            is also where currently the most work is situated hands-down,
            as each wave convolution takes maaaany sample multiplies.

            For medium freqs we will need 1st order interpolation, higher
            will need quadratic. We could max it out at cubic near nyquist.
            */

            Rng rng = new Rng(tick * 0x52EAAEBBu + (uint)p * 0x5A9CA13Bu + (uint)(centerFreq * 0xCD0445A5u) * 0xE0EB6C25u);

            float nHalf = cfg.cyclesPerWave / 2f;
            float smpPerPix = signal.Length / (float)cfg.texWidth;
            float smpPerPixHalf = (smpPerPix) * 0.5f;
            float smpPerCycle = sr / centerFreq;
            float smpPerWave = smpPerCycle * cfg.cyclesPerWave;
            float smpPerWaveInv = 1f / smpPerWave;

            int convsPerPix = (int)math.ceil(((smpPerPix / smpPerWave) * cfg.convsPerPixMultiplier));
            float convsPerPixInv = 1f / convsPerPix;
            float convStep = smpPerPix / (float)convsPerPix;

            float timeSpan = 1f / centerFreq * nHalf;

            float dotSum = 0f;

            for (int c = 0; c < convsPerPix; c++) {
                float freq = centerFreq + rng.NextFloat(-0.001f, 0.001f) * centerFreq;
                smpPerCycle = sr / freq;
                smpPerWave = smpPerCycle * cfg.cyclesPerWave;
                smpPerWaveInv = 1f / smpPerWave;

                timeSpan = 1f / freq * nHalf;

                float smpStart = p * smpPerPix + c * convStep - 0.5f * smpPerWave + rng.NextFloat(-0.005f, 0.005f) * smpPerCycle;

                float waveDot = 0f;
                for (int w = 0; w <= smpPerWave; w++) {
                    float waveTime = -timeSpan + (w * smpPerWaveInv) * (timeSpan * 2f);
                    int signalIdx = (int)(smpStart + w);

                    if (signalIdx < 0 || signalIdx >= signal.Length) {
                        continue;
                    }

                    float2 wave = WUtils.MorletComplex(waveTime, freq, cfg.cyclesPerWave);
                    wave = WUtils.CMul(wave, new float2(signal[signalIdx], 0f));

                    waveDot += wave.x;
                }

                dotSum += math.abs(waveDot) * smpPerWaveInv;
            }

            scaleogram[p] = dotSum * convsPerPixInv;
        }
    }

    [BurstCompile]
    public struct CopyRowJob : IJobParallelFor {
        [ReadOnly] public NativeArray<float> row;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> targetTex;
        [ReadOnly] public int rowIdx;
        [ReadOnly] public TransformConfig cfg;

        public void Execute(int i) {
            targetTex[(rowIdx * cfg.texWidth + i)] = row[i];
        }
    }
}
