/*
Todo:
- Nice, this learns *something*, but we can see architectural limits cap out at 36% accuracy
- We could still try sigmoid+softmax+crossentropy, but this is really where convolution needs
to happen.
 */

using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using System.Collections.Generic;
using NNBurst.Cifar;
using DataManager = NNBurst.Cifar.DataManager;
using Rng = Unity.Mathematics.Random;

namespace NNBurst {
    public class CifarTraining : MonoBehaviour {
        [SerializeField] private bool _testEachEpoch;

        Rng _rng;

        private FCNetwork _net;
        private FCGradients _gradients;
        private FCGradients _gradientsAvg;

        private NativeArray<int> _batch;
        NativeArray<float> _targetOutputs;
        NativeArray<float> _dCdO;
        NativeArray<float> _inputs;

        int _epochCount;
        int _batchCount;
        float _trainingLoss;
        float _rate;

        const int OutputClassCount = 10;
        const int BatchSize = 10;

        System.Diagnostics.Stopwatch _watch;

        Texture2D _img;
        Label _lbl;

        private void Awake() {
            Application.runInBackground = true;

            DataManager.Load();

            _rng = new Rng(1234);

            var config = new FCNetworkConfig();
            config.Layers.Add(new FCLayerConfig { Neurons = DataManager.ImgDims * DataManager.Channels });
            config.Layers.Add(new FCLayerConfig { Neurons = 40 });
            config.Layers.Add(new FCLayerConfig { Neurons = 20 });
            config.Layers.Add(new FCLayerConfig { Neurons = 10 });

            _net = new FCNetwork(config);
            NeuralUtils.Initialize(_net, ref _rng);

            _gradients = new FCGradients(config);
            _gradientsAvg = new FCGradients(config);
            
            _batch = new NativeArray<int>(BatchSize, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            _targetOutputs = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _dCdO = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _inputs = new NativeArray<float>(DataManager.ImgDims * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            _watch = System.Diagnostics.Stopwatch.StartNew();

            int testImgIdx = 8392;
            _lbl = DataManager.Test.Labels[testImgIdx];
            _img = new Texture2D(32, 32, TextureFormat.ARGB32, false, true);
            DataManager.ToTexture(DataManager.Test, testImgIdx, _img);

            Test();
        }

        private void Update() {
            if (_epochCount < 30) {
                if (_batchCount < (DataManager.Train.Labels.Length/ BatchSize)) {
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
                GUILayout.Label("Batch: " + _batchCount + "/" + (DataManager.Train.Labels.Length / BatchSize));
                GUILayout.Label("Train Loss: " + _trainingLoss);
                GUILayout.Label("Rate: " + _rate);
            }
            GUILayout.EndVertical();

            GUI.Label(new Rect(0f, 128f, 320f, 320f), "Label: " + _lbl);
            GUI.DrawTexture(new Rect(0f, 160f, 320f, 320f), _img, ScaleMode.ScaleToFit);
        }

        private void OnDestroy() {
            DataManager.Unload();

            _net.Dispose();
            _gradients.Dispose();
            _gradientsAvg.Dispose();

            _batch.Dispose();
            _targetOutputs.Dispose();
            _dCdO.Dispose();
            _inputs.Dispose();
        }

        private void TrainMinibatch() {
            UnityEngine.Profiling.Profiler.BeginSample("TrainMiniBatch");

            float avgTrainCost = 0f;

            DataManager.GetBatch(_batch, DataManager.Train, ref _rng);

            var handle = NeuralJobs.ZeroGradients(_gradientsAvg);

            for (int i = 0; i < _batch.Length; i++) {
                handle = DataManager.CopyInput(_inputs, DataManager.Train, _batch[i], handle);
                handle = NeuralJobs.ForwardPass(_net, _inputs, handle);

                int lbl = (int)DataManager.Train.Labels[_batch[i]];
                handle.Complete();
                NeuralMath.ClassToOneHot(lbl, _targetOutputs); // Todo: job

                handle = NeuralJobs.BackwardsPass(_net, _gradients, _inputs, _targetOutputs, handle);
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
            for (int i = 0; i < DataManager.Test.NumImgs; i++) {
                int lbl = (int)DataManager.Test.Labels[i];

                var handle = DataManager.CopyInput(_inputs, DataManager.Test, i);
                handle = NeuralJobs.ForwardPass(_net, _inputs, handle);
                handle.Complete();

                int predictedLbl = NeuralMath.ArgMax(_net.Last.Outputs);
                if (predictedLbl == lbl) {
                    correctTestLabels++;
                }
            }

            float accuracy = correctTestLabels / (float)DataManager.Test.NumImgs;
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