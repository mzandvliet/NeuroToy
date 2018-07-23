using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Fourier = Analysis.Fourier;
using Unity.Jobs;

/*
    Todo:
    This basic example is still not correct. FIX.
    The reconstructed signal, for fractional frequency values, has DC offset
    and phase issues. For n.5f freqs, we get a gaussian distribution for
    amplitudes in frequency space, centered around the two nearest buckets.
    Looking along the imaginary axis, we see a kind of 1/x pattern, switching
    polarity around the frequency center. Sample[0] reconstruction is no longer
    zero, but has DC offset. In fact, the whole signal is DC offset at these
    fractional frequency values.

    Get thoroughly familiar with frequency space arithmetic
    Study why forward transform rotates one way, not the other
    Play around with additive synthesis, fm
    Write tests, proofs, so you can rely on your code

    We previously had an interesting non-linear distribution of basis vectors.
    Reason about what it was, and what kind of effects that had. It's funny
    that I never bothered to check what the actual frequencies were that I
    was tuning the basis vectors to, they were way off from what it should have
    been. It says I either need to make that stuff way easier to inspect, or
    work in an environment where it is much eachier to check on all these things
    as you compute.

    So, say... Mathematica. Where all these experiments I do for study would go
    at lightspeedd compared to what I'm doing now. In Unity I know how to go fast
    and to do all sorts of custom graphics and simulation. But that's not what
    I need for these quests of understanding.
 */
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
            samples[i] = math.sin(
                Complex2f.Tau * phaseOffset +
                Complex2f.Tau * freq * phase) * amp;
            // samples[i] += math.sin(
            //     Complex2f.Tau * phaseOffset +
            //     Complex2f.Tau * freq * 2f * phase) * amp / 2f;
        }
    }

    private void DoTransform(NativeArray<float> input, NativeArray<float> output) {
        var watch = System.Diagnostics.Stopwatch.StartNew();

        var ft= new Analysis.Fourier.TransformWindowJob();
        ft.InReal = input;
        ft.OutSpectrum = _spectrum;
        ft.Samplerate = _sr;
        ft.WindowSize = input.Length;
        ft.WindowStart = 0;
        var h = ft.Schedule();

        var ift = new Analysis.Fourier.InverseTransformWindowJob();
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
        Gizmos.color = Color.white;
        DrawSignal(_inputSignal);
        Gizmos.color = Color.magenta;
        DrawSignal(_outputSignal);

        // Spectrum
        float specXScale = 0.1f;
        for (int i = 0; i < FreqBins; i++) {
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(
                new Vector3(i * specXScale + 0.01f, 0f, 0f),
                Vector3.forward * math.length(_spectrum[i]) * _renderScale);

            Gizmos.color = Color.red;
            Gizmos.DrawRay(
                new Vector3(i * specXScale, 0f, 0f),
                new Vector3(0f, _spectrum[i].x * _renderScale, _spectrum[i].y * _renderScale));

            Gizmos.DrawSphere(new Vector3(i * specXScale, _spectrum[i].x * _renderScale, _spectrum[i].y * _renderScale), 0.01f);
        }
    }

    private static void DrawSignal(NativeArray<float> signal) {
        float sigXScale = 0.05f;
        for (int i = 0; i < signal.Length; i++) {
            Gizmos.DrawRay(new Vector3(i * sigXScale, 0f, 2f), new Vector3(0f,  signal[i], 0f));
        }
        for (int i = 1; i < signal.Length; i++) {
            Gizmos.DrawLine(new Vector3((i-1) * sigXScale, signal[i-1], 2f), new Vector3(i * sigXScale,  signal[i], 2f));
        }
    }
}