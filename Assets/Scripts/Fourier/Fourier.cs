using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;

/*
Todo: Rewrite using Burst jobs and Unity Mathematics

- Can parallelize over:
 - Windows
 - Frequency bins within windows
 - Individual samples within windows, if needed

- Implement FFT after vanilla
- Keep benchmarks around

- Then try regression techniques for finding sparse fourier transforms that work best for certain classes of signal.
 */

namespace Analysis {
    public static class Fourier {
        public static void Transform(NativeArray<float> input, List<NativeArray<float2>> spectra, int windowSize, int sr) {
            for (int i = 0; i < spectra.Count; i++) {
                var job = new TransformWindowJob();
                job.InReal = input;
                job.OutSpectrum = spectra[i];
                job.WindowStart = i * windowSize;
                job.WindowSize = windowSize;
                job.Samplerate = sr;
                var h = job.Schedule();

                h.Complete();
            }
        }

        public static void InverseTransform(List<NativeArray<float2>> spectra, NativeArray<float> output, int windowSize, int sr) {
            for (int i = 0; i < spectra.Count; i++) {
                var job = new InverseTransformWindowJob();
                job.OutReal = output;
                job.InSpectrum = spectra[i];
                job.WindowStart = i * windowSize;
                job.WindowSize = windowSize;
                var h = job.Schedule();
                h.Complete();
            }
        }

        [BurstCompile]
        public struct TransformWindowJob : IJob {
            [ReadOnly] public NativeArray<float> InReal;
            public NativeArray<float2> OutSpectrum;
            [ReadOnly] public int WindowStart;
            [ReadOnly] public int WindowSize;
            [ReadOnly] public int Samplerate;

            public void Execute() {
                if (WindowSize == 0) {
                    throw new System.ArgumentException("Window Size cannot be zero");
                }
                if (OutSpectrum.Length > Samplerate / 2) {
                    throw new System.ArgumentException(string.Format("Amount of frequency bins, {0}, cannot exceed Nyquist limit, {1}/2={2}", OutSpectrum.Length, Samplerate, Samplerate/2));
                }

                for (int k = 0; k < OutSpectrum.Length; k++) {
                    OutSpectrum[k] = new float2();

                    float freq = k;
                    //Debug.Log("k: " + k + ", freq: " + freq);

                    // float phaseStep = -Complex2f.Tau * freq / sr;
                    // float2 rotor = new float2(
                    //     math.cos(phaseStep),
                    //     math.sin(phaseStep));
                    // float2 phase = new float2(1f, 0f);

                    // for (int s = 0; s < windowSize; s++) {
                    //     spectrum[k] += Complex2f.Mul(new float2(signal[windowStart + s], 0f), phase);
                    //     phase = Complex2f.Mul(phase, rotor);
                    // }

                    float step = -Complex2f.Tau * k / WindowSize;
                    for (int s = 0; s < WindowSize; s++) {
                        float2 v = new float2(
                            InReal[WindowStart + s] * math.cos(step * s), // + InImag * math.sin(step * s)
                            InReal[WindowStart + s] * math.sin(step * s) // + InImag * math.cos(step * s)
                        );
                        OutSpectrum[k] += v;
                    }

                    OutSpectrum[k] /= OutSpectrum.Length;
                }
            }
        }

        [BurstCompile]
        public struct InverseTransformWindowJob : IJob {
            [ReadOnly] public NativeArray<float2> InSpectrum;
            public NativeArray<float> OutReal;
            [ReadOnly] public int WindowStart;
            [ReadOnly] public int WindowSize;

            public void Execute() {
                if (WindowSize == 0) {
                    throw new System.ArgumentException("Window Size cannot be zero");
                }

                for (int k = 0; k < InSpectrum.Length; k++) {
                    // float freq = k;
                    // float phaseStep = Complex2f.Tau * freq / sr;
                    // var rotor = new float2(
                    //     math.cos(phaseStep),
                    //     math.sin(phaseStep));
                    // var phase = new float2(1f, 0f);

                    // for (int s = 0; s < sr; s++) {
                    //     signal[s] += Complex2f.Mul(spectrum[k], phase).x;
                    //     phase = Complex2f.Mul(phase, rotor);
                    // }

                    float step = Complex2f.Tau * k / WindowSize;
                    for (int s = 0; s < WindowSize; s++) {
                        float2 v = new float2(
                            OutReal[WindowStart + s] += math.cos(step * s) * InSpectrum[k].x,
                            OutReal[WindowStart + s] += math.sin(step * s) * InSpectrum[k].y

                            //OutImag[WindowStart + s] += math.sin(step * s) * InSpectrum[k].x,
                            //OutImag[WindowStart + s] += math.cos(step * s) * InSpectrum[k].y,
                        );
                    }
                }
            }
        }

        // Todo: Can also store all windows of a signal in a nativearray, use slicing
        public static void Filter(List<NativeArray<float2>> spectrum) {
            int freqResolution = spectrum[0].Length;
            int halfFreqRes = freqResolution / 2;

            for (int w = 0; w < spectrum.Count; w++) {
                Filter(spectrum[w]);
            }
        }

        public static void Filter(NativeArray<float2> spectrum) {
            for (int i = 0; i < spectrum.Length; i++) {
                spectrum[i] *= 2f;
            }
        }

        public static Vector2 RandomOnUnitCircle() {
            float phase = Random.Range(0, Mathf.PI * 2);
            return new Vector2(
                math.cos(phase),
                math.sin(phase));
        }

        public static void Copy(NativeArray<float> signal, int start, NativeArray<float> window) {
            if (start + window.Length > signal.Length) {
                Debug.LogError("Requested window extends beyond signal duration");
                return;
            }

            for (int i = 0; i < window.Length; i++) {
                window[i] = signal[start + i];
            }
        }

        public static void Copy(NativeArray<float> a, float[] b) {
            if (a.Length != b.Length) {
                Debug.LogError("Copy needs input arrays to be of same size");
                return;
            }

            for (int i = 0; i < a.Length; i++) {
                b[i] = a[i];
            }
        }

        public static void Copy(float[] a, NativeArray<float> b) {
            if (a.Length != b.Length) {
                Debug.LogError("Copy needs input arrays to be of same size");
                return;
            }

            for (int i = 0; i < a.Length; i++) {
                b[i] = a[i];
            }
        }

        public static void CopyChannel(float[] interleaved, NativeArray<float> output, int numChannels, int channel) {
            for (int i = 0; i < output.Length; i++) {
                output[i] = interleaved[i * numChannels + channel];
            }
        }

        public static void Clear(NativeArray<float> buffer) {
            for (int i = 0; i < buffer.Length; i++) {
                buffer[i] = 0f;
            }
        }

        public static void Clear(float[] buffer) {
            for (int i = 0; i < buffer.Length; i++) {
                buffer[i] = 0f;
            }
        }

        public static List<NativeArray<float2>> Allocate(int signalLength, int windowSize, int freqResolution) {
            int windows = Mathf.FloorToInt(signalLength / windowSize); // Todo: last fractional window!
            var spectra = new List<NativeArray<float2>>();
            for (int i = 0; i < windows; i++) {
                var spectrum = new NativeArray<float2>(freqResolution, Allocator.Persistent, NativeArrayOptions.ClearMemory);
                spectra.Add(spectrum);
            }
            return spectra;
        }

        public static List<NativeArray<float2>> Deallocate(List<NativeArray<float2>> spectra) {
            for (int i = 0; i < spectra.Count; i++) {
                spectra[i].Dispose();
            }
            spectra.Clear();
            return spectra;
        }

        public static void DrawSignal(NativeArray<float> signal, float xScale, float yScale) {
            // for (int i = 0; i < signal.Length; i++) {
            //     Gizmos.DrawRay(
            //         new Vector3(i * xScale, 0f, 2f),
            //         new Vector3(0f, signal[i] * yScale, 0f));
            // }
            for (int i = 1; i < signal.Length; i++) {
                Gizmos.DrawLine(
                    new Vector3((i - 1) * xScale, signal[i - 1] * yScale, 2f),
                    new Vector3(i * xScale, signal[i] * yScale, 2f));
            }
        }

        public static void DrawSpectrum(NativeArray<float2> spectrum, float xScale, float yScale) {
            for (int i = 0; i < spectrum.Length; i++) {
                Gizmos.color = Color.blue;
                Gizmos.DrawRay(
                    new Vector3(i * xScale + 0.01f, 0f, 0f),
                    Vector3.forward * math.length(spectrum[i]) * yScale);

                Gizmos.color = Color.red;
                Gizmos.DrawRay(
                    new Vector3(i * xScale, 0f, 0f),
                    new Vector3(0f, spectrum[i].x * yScale, spectrum[i].y * yScale));

                Gizmos.DrawSphere(new Vector3(i * xScale, spectrum[i].x * yScale, spectrum[i].y * yScale), 0.01f);
            }
        }
    }
}
