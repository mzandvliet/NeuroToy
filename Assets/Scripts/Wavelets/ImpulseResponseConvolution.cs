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

public class ImpulseResponseConvolution : MonoBehaviour
{
    [SerializeField] private AudioClip _clipSignal;
    [SerializeField] private AudioClip _clipIR;

    private NativeArray<float> _signal;
    private NativeArray<float> _ir;
    private NativeArray<float> _output;

    private AudioClip _clipOutput;
    private AudioSource _source;

    private void Awake() {
        int sr = _clipSignal.frequency;

        _signal = LoadClip(_clipSignal, Allocator.Persistent);
        _ir = LoadClip(_clipIR, Allocator.Persistent);
        _output = new NativeArray<float>(_signal.Length + _ir.Length, Allocator.Persistent);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        Convolve(_signal, _ir, _output);
        watch.Stop();
        Debug.LogFormat("Total time: {0} ms", watch.ElapsedMilliseconds);

        _clipOutput = AudioClip.Create("ConvolvedResult", _output.Length, 1, _clipSignal.frequency, false);
        var outputManaged = new float[_output.Length];
        for (int i = 0; i < _output.Length; i++)
        {
            outputManaged[i] = _output[i];
        }
        _clipOutput.SetData(outputManaged, 0);

        _source = gameObject.AddComponent<AudioSource>();
        _source.clip = _clipOutput;

        var wavPath = System.IO.Path.Combine(Application.dataPath, string.Format("{0}.wav", System.DateTime.Now.ToFileTimeUtc()));
        SavWav.Save(wavPath, _clipOutput);
    }

    private void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            _source.Play();
        }
    }

    private static NativeArray<float> LoadClip(AudioClip clip, Allocator allocator) {
        var buffer = new NativeArray<float>(clip.samples, allocator);

        var data = new float[clip.samples];
        clip.GetData(data, 0);
        for (int i = 0; i < data.Length; i++) {
            buffer[i] = data[i];
        }

        return buffer;
    }

    private void OnDestroy() {
        _signal.Dispose();
        _ir.Dispose();
        _output.Dispose();
    }

    private void Convolve(NativeArray<float> signal, NativeArray<float> ir, NativeArray<float> output) {
        var job = new ConvolveJob() {
            signal = signal,
            ir = ir,
            output = output
        };
        job.Schedule().Complete();
    }

    [BurstCompile]
    public struct ConvolveJob : IJob {
        [ReadOnly] public NativeSlice<float> signal;
        [ReadOnly] public NativeSlice<float> ir;

        public NativeSlice<float> output;
        
        public void Execute() {
            float maxAmp = float.Epsilon;

            for (int i = 0; i < signal.Length; i++)
            {
                for (int j = 0; j < ir.Length; j++)
                {
                    output[i + j] += signal[i] * ir[j];
                }

                // output[i] = math.sin((i / 44100f) * (math.PI * 2f * 440f));

                float absAmp = math.abs(output[i]);
                if (absAmp > maxAmp) {
                    maxAmp = absAmp;
                }
            }

            // float normInv = 1f / (float)ir.Length;
            float normInv = 1f / maxAmp;

            for (int i = 0; i < output.Length; i++) {
                output[i] *= normInv;
            }
        }
    }
}
