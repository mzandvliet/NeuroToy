﻿using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Application = UnityEngine.Application;
using Debug = UnityEngine.Debug;
using Fourier = Analysis.Fourier;

/*
 * Todo:
 * - Investigate difference in quality between current transcendental and complex versions
 *      - Transcendental has very noticable discontinuities at window borders
 *      - Somehow, complex version does not have these. Why?
 *      - Has to do with sign of step value
 * - STFT with overlapping windows
 *
 * Performance
 * - Further jobbification
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

    private NativeArray<float> _input;
    private NativeArray<float> _output;
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
        _input = new NativeArray<float>(samples.Length / clip.channels, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        _output = new NativeArray<float>(samples.Length / clip.channels, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        Fourier.CopyChannel(samples, _input, clip.channels, 0);

        _spectra = Fourier.Allocate(_input.Length, WindowSize, FreqBins);
        var watch = Stopwatch.StartNew();
        Fourier.Transform(_input, _spectra, WindowSize, _sr);
        //Fourier.Filter(_spectra);
        Fourier.InverseTransform(_spectra, _output, WindowSize, _sr);
        watch.Stop();
        Debug.Log("FT+IFT millis:" + watch.ElapsedMilliseconds);

        var outputSamples = new float[_output.Length];
        Fourier.Copy(_output, outputSamples);

        // Play result
        _result = AudioClip.Create("Result", outputSamples.Length, 1, _sr, false);
        _result.SetData(outputSamples, 0);
        _source.clip = _result;
        _source.Play();
    }

    private void OnDestroy() {
        _input.Dispose();
        _output.Dispose();
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

        const float xScale = 0.002f;
        const float yScale = 50f;
        Gizmos.color = Color.white;
        Fourier.DrawSignal(_input, Vector3.forward, xScale, yScale);
        Gizmos.color = Color.magenta;
        Fourier.DrawSignal(_output, Vector3.zero, xScale, yScale);

        const float spectrumXScale = 0.05f;
        const float spectrumYScale = 1f;
        Fourier.DrawSpectrum(_spectra[_spectra.Count/2], Vector3.zero, spectrumXScale, spectrumYScale);
    }

    private void OnGUI() {
        GUILayout.BeginVertical(GUI.skin.box);
        {
            GUILayout.Label("Window: " + _currentWindow);
        }
        GUILayout.EndVertical();
    }
}
