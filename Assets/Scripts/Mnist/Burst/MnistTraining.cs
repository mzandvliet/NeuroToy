/*
Todo:
- Continue the trend of definining computation graphs beyond the BurstJob level.
 */

using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using System.Collections.Generic;
using NNBurst.Mnist;
using Rng = Unity.Mathematics.Random;

namespace NNBurst {
    public class MnistTraining : MonoBehaviour {
        [SerializeField] private bool _testEachEpoch;

        Rng _rng;

        private FCNetwork _net;
        private FCGradients _gradients;
        private FCGradients _gradientsAvg;

        private NativeArray<int> _batch;
        NativeArray<float> _targetOutputs;
        NativeArray<float> _dCdO;

        int _epochCount;
        int _batchCount;
        float _trainingLoss;
        float _rate;

        const int OutputClassCount = 10;
        const int BatchSize = 10;

        System.Diagnostics.Stopwatch _watch;

        // Visual test of data
        private int _label;
        private Texture2D _tex;

        private void Awake() {
            Application.runInBackground = true;

            DataManager.LoadFloatData();

            _rng = new Rng(1234);

            var config = new FCNetworkConfig();
            config.Layers.Add(new FCLayerConfig { NumNeurons = DataManager.ImgDims });
            config.Layers.Add(new FCLayerConfig { NumNeurons = 30 });
            config.Layers.Add(new FCLayerConfig { NumNeurons = 10 });

            _net = new FCNetwork(config);
            NeuralUtils.Initialize(_net, ref _rng);

            _gradients = new FCGradients(config);
            _gradientsAvg = new FCGradients(config);
            _batch = new NativeArray<int>(BatchSize, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            _targetOutputs = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _dCdO = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            _watch = System.Diagnostics.Stopwatch.StartNew();

            // Visual test of data
            const int testImgIdx = 18;
            _label = DataManager.TrainFloats.Labels[testImgIdx];
            _tex = new Texture2D(DataManager.Width, DataManager.Height, TextureFormat.ARGB32, false, true);
            _tex.filterMode = FilterMode.Point;
            DataManager.ToTexture(DataManager.TrainFloats, testImgIdx, _tex);

            Test();
        }

        private void Update() {
            if (_epochCount < 30) {
                if (_batchCount < DataManager.TrainFloats.Labels.Length/BatchSize) {
                    for (int i = 0; i < 100; i++) {
                        TrainMinibatch();
                    }
                } else {
                    if (_testEachEpoch) {
                        Test();
                    }
                    _batchCount = 0;
                    _epochCount++;
                }
            } else {
                Test();
                _watch.Stop();
                Debug.Log("Time taken: " + System.Math.Round(_watch.ElapsedMilliseconds / 1000.0) + " seconds");
                gameObject.SetActive(false);
            }
        }

        private void OnGUI() {
            GUILayout.BeginVertical(GUI.skin.box);
            {
                GUILayout.Label("Epoch: " + _epochCount);
                GUILayout.Label("Batch: " + _batchCount + "/" + (DataManager.TrainFloats.Labels.Length / BatchSize));
                GUILayout.Label("Train Loss: " + _trainingLoss);
                GUILayout.Label("Rate: " + _rate);
            }
            GUILayout.EndVertical();

            GUI.Label(new Rect(0f, 128f, 280f, 32f), "Label: " + _label);
            GUI.DrawTexture(new Rect(0f, 148f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
        }

        private void OnDestroy() {
            DataManager.Unload();

            _net.Dispose();
            _gradients.Dispose();
            _gradientsAvg.Dispose();

            _batch.Dispose();
            _targetOutputs.Dispose();
            _dCdO.Dispose();
        }

        private void TrainMinibatch() {
            UnityEngine.Profiling.Profiler.BeginSample("TrainMiniBatch");

            float avgTrainCost = 0f;

            DataManager.GetBatch(_batch, DataManager.TrainFloats, ref _rng);

            var handle = NeuralJobs.ZeroGradients(_gradientsAvg);

            for (int i = 0; i < _batch.Length; i++) {
                handle = DataManager.CopyInput(_net.Inputs, DataManager.TrainFloats, _batch[i], handle);
                handle = NeuralJobs.ForwardPass(_net, handle);

                int lbl = DataManager.TrainFloats.Labels[_batch[i]];
                handle.Complete();
                NeuralMath.ClassToOneHot(lbl, _targetOutputs); // Todo: job

                handle = NeuralJobs.BackwardsPass(_net, _gradients, _targetOutputs, handle);
                handle = NeuralJobs.AddGradients(_gradients, _gradientsAvg, handle);
                handle.Complete();

                // Todo: backwards pass logic now does this, don't redo, just check
                NeuralMath.Subtract(_targetOutputs, _net.Last.Outputs, _dCdO);
                float cost = NeuralMath.Cost(_dCdO);
                avgTrainCost += cost;
            }

            // Update weights and biases according to averaged gradient and learning rate
            _rate = 3.0f / (float)BatchSize;
            handle = NeuralJobs.UpdateParameters(_net, _gradientsAvg, _rate, handle);
            handle.Complete(); // Todo: Is this one needed?

            _batchCount++;

            avgTrainCost /= (float)BatchSize;
            _trainingLoss = (float)System.Math.Round(avgTrainCost, 6);

            UnityEngine.Profiling.Profiler.EndSample();
        }

        private void Test() {
            UnityEngine.Profiling.Profiler.BeginSample("Test");

            int correctTestLabels = 0;
            for (int i = 0; i < DataManager.TestFloats.NumImgs; i++) {
                int lbl = DataManager.TestFloats.Labels[i];

                var handle = DataManager.CopyInput(_net.Inputs, DataManager.TestFloats, i);
                handle = NeuralJobs.ForwardPass(_net, handle);
                handle.Complete();

                int predictedLbl = NeuralMath.ArgMax(_net.Last.Outputs);
                if (predictedLbl == lbl) {
                    correctTestLabels++;
                }
            }

            float accuracy = correctTestLabels / (float)DataManager.TestFloats.NumImgs;
            Debug.Log("Test Accuracy: " + System.Math.Round(accuracy * 100f, 4) + "%");

            UnityEngine.Profiling.Profiler.EndSample();
        }

        private static void Print(NativeArray<float> a) {
            string s = "[";
            for (int i = 0; i < a.Length; i++) {
                s += a[i] + ", ";
            }
            s += "]";
            Debug.Log(s);
        }
    }
}