using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Application = UnityEngine.Application;
using Debug = UnityEngine.Debug;
using Fourier = Analysis.Fourier;

/*
 * Todo:
 *
 * Performance
 * - Without thread sync, while this works, the drawn spectrum is changed at random times while rendering it, causing visual discontinuity
 * - With naive locking we hold up the audio thread unnecessarily.
 * - Use producer/consumer system without locking
 * - Use job system and compute shaders
 * - FFT (radix2 or decimation in time)
 * 
 * - Store results as an image or a mesh, so we can see many consecutive frames
 */

public class FourierAudioAnalysis : MonoBehaviour {
    [SerializeField] private AudioSource _source;
    [SerializeField] private float _renderScale = 1f;

    private const int WindowSize = 512;
    private const int FreqBins = 256;
    private int _sr;

    private List<NativeArray<float2>> _spectra;
    private int _currentWindow;

    
    private AudioClip _result;

    void Start() {
        Application.runInBackground = true;
        _source = gameObject.GetComponent<AudioSource>();
        _sr = 44100;

        // Get left channel data from clip
        var clip = _source.clip;
        var samples = new float[clip.samples];
        clip.GetData(samples, 0);
        var samplesNative = new NativeArray<float>(samples.Length / clip.channels, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        Fourier.CopyChannel(samples, samplesNative, clip.channels, 0);

        _spectra = Fourier.Allocate(samplesNative.Length, WindowSize, FreqBins);
        var watch = Stopwatch.StartNew();
        Fourier.Transform(samplesNative, _spectra, WindowSize, _sr);
        Fourier.InverseTransform(_spectra, samplesNative, WindowSize, _sr);
        watch.Stop();
        Debug.Log("FT+IFT millis:" + watch.ElapsedMilliseconds);

        var outputSamples = new float[samplesNative.Length];
        Fourier.Copy(samplesNative, outputSamples);

        // Play result
        _result = AudioClip.Create("Result", outputSamples.Length, 1, _sr, false);
        _result.SetData(outputSamples, 0);
        _source.clip = _result;
        _source.Play();

        samplesNative.Dispose();
    }

    private void OnDestroy() {
        Fourier.Deallocate(_spectra);
    }

    private void Update() {
        if (Input.GetKeyDown(KeyCode.A)) {
            ScrollWindow(-1);
        }
        if (Input.GetKeyDown(KeyCode.D)) {
            ScrollWindow(+1);
        }
    }

    private void ScrollWindow(int offset) {
        _currentWindow = Mathf.Clamp(_currentWindow + offset, 0, _spectra.Count - 1);
    }

    private void OnDrawGizmos() {
        if (!Application.isPlaying || _spectra == null) {
            return;
        }

        for (int i = 0; i < FreqBins; i++) {
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(
                new Vector3(i * 0.01f, 0f, 0f),
                 Vector3.up * (math.abs(math.lengthSquared(_spectra[_currentWindow][i])) * _renderScale));
        }
    }

    private void OnGUI() {
        GUILayout.BeginVertical(GUI.skin.box);
        {
            GUILayout.Label("Window: " + _currentWindow);
        }
        GUILayout.EndVertical();
    }
}

public static class Test {
    /* 
     * This tests the difference in phase for two different methods of computing
     * the wave patterns for each Fourier resonator, which is to be convolved
     * with the signal. Method 1 uses Sin/Cos evaluation, method 2 evolves
     * a stateful complex oscillator.
     * 
     * Method 2 is twice as fast as method 1 for a full transform, with hardly
     * any error to worry about. In almost all cases it works out to the exact
     * same bit representation. Slight errors for some bands are symmetric for
     * the forward and backward transform, so are at least internally consistent.
     * 
     * Error pattern seems constant, regardless of choice of freqResolution and
     * windowLength.
     */
    public static void MeasureAnalyticsVsStatePhasorError() {
        const int freqBins = 512;
        const int windowLength = 256;
        const int sr = 44100;
       
        int nyquist = sr / 2;
        int kScale = nyquist / freqBins;

        float maxPhaseDiff = 0f;
        int maxPhaseDiffK = 0;

        for (int k = 0; k < freqBins; k++) {
            float scale = k * kScale;
            float phaseStep = -Mathf.PI * 2f * scale / sr;

            Vector2 phaseStepper = new Vector2(
                Mathf.Cos(phaseStep),
                Mathf.Sin(phaseStep));

            Vector2 statePhasor = new Vector2(1f, 0f);
            float phaseDiff = 0f;

            for (int s = 0; s < windowLength; s++) {
                float phase = s * phaseStep;

                Vector2 analyticPhasor = new Vector2(
                    Mathf.Cos(phase),
                    Mathf.Sin(phase));

                phaseDiff = Vector2.Angle(analyticPhasor, statePhasor);

                statePhasor = Complex2f.Mul(statePhasor, phaseStepper);
            }

            Debug.Log("k: " + k + ", Phase difference: " + phaseDiff);

            if (phaseDiff > maxPhaseDiff) {
                maxPhaseDiff = phaseDiff;
                maxPhaseDiffK = k;
            }
        }

        Debug.Log("Max Phase Diff: " + maxPhaseDiff + " at k = " + maxPhaseDiffK);
    }

    /* This checks that our arithmetic for generating frequency bin results
     * in a symmetric distribution over positive and negative frequency space
     * Todo: lol, this is so wrong
     */
    public static void CreateFreqBins() {
        int freqBins = 32;
        int sr = 44100;
        int nyquist = sr / 2;

        float kScale = nyquist / (float)(freqBins-1);

        for (int k = 0; k < freqBins; k++) {
            float scale = -nyquist + k * kScale * 2f;
            Debug.Log("k: " + k + ", scale: " + scale);
        }

        Debug.Log(-nyquist);
        Debug.Log(-nyquist + (freqBins-1) * kScale * 2f);
    }
}