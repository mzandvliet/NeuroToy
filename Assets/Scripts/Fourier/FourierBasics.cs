using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Fourier = Analysis.Fourier;
using Unity.Jobs;

public class FourierBasics : MonoBehaviour {
    [SerializeField] private float _renderScale = 1f;
    [SerializeField] private AudioSource _source;

    private static int FreqBins = 512;
    private int _sr;

    private NativeArray<float2> _spectrum;
    private NativeArray<float> _inputSignal;
    private NativeArray<float> _outputSignal;

    private float _freq = 1f;
    private float _phase = 0f;

    private AudioClip _result;

    void Start() {
        Application.runInBackground = true;
        _sr = 1024;

        FreqBins = Mathf.Min(FreqBins, _sr / 2);

        _inputSignal = new NativeArray<float>(_sr, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        _outputSignal = new NativeArray<float>(_sr, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        _spectrum = new NativeArray<float2>(FreqBins, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        
        GenerateSignal(_inputSignal, _freq, 0f, 1f, _sr);
        DoTransform(_inputSignal, _outputSignal);
    }

    private void OnDestroy() {
        _inputSignal.Dispose();
        _outputSignal.Dispose();
        _spectrum.Dispose();
    }

    private void Update() {
        bool update = false;
        // Change freq
        if (Input.GetKey(KeyCode.S)) {
            _freq = Mathf.Clamp(_freq - 1f * Time.deltaTime, 0, _sr/2);
            update = true;
        } else if (Input.GetKey(KeyCode.W)) {
            _freq = Mathf.Clamp(_freq + 1f * Time.deltaTime, 0, _sr/2);
            update = true;
        }
        // Change phase
        if (Input.GetKeyDown(KeyCode.A)) {
            _phase = Mathf.Clamp(_phase + 0.1f, - 1f/_freq, 1f/_freq);
            update = true;
        } else if (Input.GetKeyDown(KeyCode.D)) {
            _phase = Mathf.Clamp(_phase - 0.1f, - 1f/_freq, 1f/_freq);
            update = true;
        }

        if (update) {
            Fourier.Clear(_inputSignal);
            Fourier.Clear(_outputSignal);
            GenerateSignal(_inputSignal, _freq, _phase, 1f, _sr);
            DoTransform(_inputSignal, _outputSignal);
            
            Debug.Log("Freq: " + _freq + ", Phase: " + _phase);
        }
    }

    // Todo: job
    private static void GenerateSignal(NativeArray<float> samples, float freq, float phaseOffset, float amp, int sr) {
        for (int i = 0; i < samples.Length; i++) {
            float phase = (i / (float)(sr));
            for (float octave = 1f; octave < 5f; octave +=1f) {
                samples[i] += math.sin(
                    Complex2f.Tau * phaseOffset +
                    Complex2f.Tau * (freq * octave) * phase) * (amp / octave);
            }
        }
    }

    private void DoTransform(NativeArray<float> input, NativeArray<float> output) {
        var watch = System.Diagnostics.Stopwatch.StartNew();

        var ft = new Analysis.Fourier.FTJob();
        var ift = new Analysis.Fourier.IFTJob();
        // var ift = new Analysis.Fourier.IFTComplexJob();
        // var ft = new Analysis.Fourier.FTComplexJob();

        ft.InReal = input;
        ft.OutSpectrum = _spectrum;
        ft.Samplerate = _sr;
        ft.WindowSize = input.Length;
        ft.WindowStart = 0;
        var h = ft.Schedule();
        
        ift.InSpectrum = _spectrum;
        ift.OutReal = output;
        ift.WindowSize = output.Length;
        ift.WindowStart = 0;
        h = ift.Schedule(h);

        h.Complete();

        watch.Stop();
        Debug.Log("FT+IFT millis:" + watch.ElapsedMilliseconds);
    }

    private void OnDrawGizmos() {
        if (!Application.isPlaying) {
            return;
        }

        // Axes
        Gizmos.color = Color.white;
        Gizmos.DrawRay(new Vector3(0f, 0f, 0f), Vector3.forward);
        Gizmos.color = Color.white;
        Gizmos.DrawRay(new Vector3(0f, 0f, 0f), Vector3.up);
        Gizmos.color = Color.white;
        Gizmos.DrawRay(new Vector3(0f, 0f, 0f), Vector3.right * 100f);

        // Signals
        const float signalXScale = 0.05f;
        const float signalYScale = 1f;
        Gizmos.color = Color.white;
        Fourier.DrawSignal(_inputSignal, Vector3.forward, signalXScale, signalYScale);
        Gizmos.color = Color.magenta;
        Fourier.DrawSignal(_outputSignal, Vector3.zero, signalXScale, signalYScale);

        // Spectrum
        const float spectrumXScale = 0.05f;
        const float spectrumYScale = 1f;
        Fourier.DrawSpectrum(_spectrum, Vector3.zero, spectrumXScale, spectrumYScale);
    }
}