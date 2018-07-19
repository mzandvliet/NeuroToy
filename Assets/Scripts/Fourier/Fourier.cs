using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;

/*
Todo: Rewrite using Burst jobs and Unity Mathematics
 */

namespace Analysis {
    public static class Fourier {
        public static void Transform(NativeArray<float> signal, List<NativeArray<float2>> spectra, int windowSize, int sr) {
            var window = new NativeArray<float>(signal.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < 1; i++) { // spectra.Count
                Copy(signal, i * windowSize, window);
                TransformWindow(window, spectra[i], sr);
            }

            window.Dispose();
        }

        public static NativeArray<float2> TransformWindow(NativeArray<float> signal, NativeArray<float2> spectrum, int sr) {
            if (spectrum.Length > sr/2) {
                throw new System.ArgumentException("Amount of frequency bins cannot exceed Nyquist limit");
            }

            for (int i = 0; i < spectrum.Length; i++) {
                spectrum[i] = new float2();
            }

            for (int k = 0; k < spectrum.Length; k++) {
                float freq = k;
                //Debug.Log("k: " + k + ", freq: " + freq);
                float phaseStep = -Complex2f.Tau * freq / sr;
                float2 phaseStepper = new float2(
                    math.cos(phaseStep),
                    math.sin(phaseStep));
                float2 phase = new float2(1f, 0f);

                for (int s = 0; s < signal.Length; s++) {
                    spectrum[k] += Complex2f.Mul(new float2(signal[s], 0f), phase);
                    phase = Complex2f.Mul(phase, phaseStepper);
                }
            }

            for (int k = 0; k < spectrum.Length; k++) {
                spectrum[k] /= signal.Length;
            }

            return spectrum;
        }

        public static void InverseTransform(List<NativeArray<float2>> spectra, NativeArray<float> signal, int windowSize, int sr) {
            var window = new NativeArray<float>(windowSize, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < 1; i++) { // spectra.Count
                Clear(window);
                InverseTransformWindow(spectra[i], window);

                for (int j = 0; j < window.Length; j++) {
                    signal[i * windowSize + j] = window[j];
                }
            }

            window.Dispose();
        }

        public static void InverseTransformWindow(NativeArray<float2> spectrum, NativeArray<float> signal) {
            int freqBins = spectrum.Length;
            int sr = signal.Length;
            if (freqBins > sr/2) {
                throw new System.ArgumentException("Amount of frequency bins cannot exceed Nyquist limit");
            }

            for (int k = 0; k < freqBins; k++) {
                float freq = k;
                float phaseStep = Complex2f.Tau * freq / sr;
                var phaseStepper = new float2(
                    Mathf.Cos(phaseStep),
                    Mathf.Sin(phaseStep));
                var phase = new float2(1f, 0f);

                for (int s = 0; s < sr; s++) {
                    signal[s] += Complex2f.Mul(spectrum[k], phase).x; // Todo: replace
                    phase = Complex2f.Mul(phase, phaseStepper);
                }
            }

            for (int s = 0; s < signal.Length; s++) {
                signal[s] /= freqBins;
            }
        }

        // Todo: Can also store all windows of a signal in a nativearray
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
                Mathf.Cos(phase),
                Mathf.Sin(phase));
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

        public static List<NativeArray<float2>> Allocate(int signalLength, int windowSize, int freqResolution) {
            int windows = Mathf.FloorToInt(signalLength / windowSize); // Todo: last fractional window!
            var spectra = new List<NativeArray<float2>>();
            for (int i = 0; i < windows; i++) {
                var spectrum = new NativeArray<float2>(freqResolution, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
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
    }
}
