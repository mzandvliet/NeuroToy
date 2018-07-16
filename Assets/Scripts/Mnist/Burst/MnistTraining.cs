/*

Let's see:

- Make a list of data structures that belong to
    - a neural net
    - an optimizer
    - a training example

As a delta from our prior code that means:
- Neural net doesn't need memory for input layer allocated
    - Have first hidden layer read directly from the training data memory
    - No more need for a memcopy
- A neural net shouldn't internally store memory needed for backwards pass error/gradients
    - Should be stored in an object representing the optimizer
    - The optimizer can then preallocate all memory it needs to function
    - Optimizer could take in the NetDefinition to create correct configuration
- Our ILayer concept adds a lot of junk and obfuscates the fact that it's all just arrays of floats

So then the question becomes:
- Is it better to write complex jobs for forwards and backwards passes
- Or to compose them from more atomic jobs like Add/Hadamard/Dot?

We know that Burst is optimized for packing lots of small jobs together in time and space,
so composing from atomic jobs might not be a bad idea. It also means we get to write
our algorithms out of those blocks, making experimentation easier.

Todo:
A matrix multiply for dot producting all inputs with all weights looks like the following:
tranpose(weights) * input

How should we implement and use the transpose operation? The data remains the same, you're
just switching rows and columns.

We don't want to manually write out jobs like this all the time, but rather specify
a computation graph with some notation and have a system that builds the jobs for us.

 */

using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using System.Collections.Generic;
using NNBurst.Mnist;

namespace NNBurst {
    public class MnistTraining : MonoBehaviour {
        [SerializeField] private bool _testEachEpoch;

        System.Random _random;

        private NativeNetwork _net;
        private NativeGradients _gradients;
        private NativeGradients _gradientsAvg;

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

        private void Awake() {
            Application.runInBackground = true;

            DataManager.Load();

            _random = new System.Random();

            var config = new NativeNetworkConfig();
            config.Layers.Add(new NativeLayerConfig { Neurons = DataManager.Train.ImgDims });
            config.Layers.Add(new NativeLayerConfig { Neurons = 30 });
            config.Layers.Add(new NativeLayerConfig { Neurons = 10 });

            _net = new NativeNetwork(config);
            NeuralUtils.Initialize(_net, _random);

            _gradients = new NativeGradients(config);
            _gradientsAvg = new NativeGradients(config);
            _batch = new NativeArray<int>(BatchSize, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            _targetOutputs = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _dCdO = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            _inputs = new NativeArray<float>(DataManager.Test.ImgDims, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            _watch = System.Diagnostics.Stopwatch.StartNew();
        }

        private void Update() {
            if (_epochCount < 30) {
                if (_batchCount < DataManager.Train.Labels.Length/BatchSize) {
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

            // GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _label);
            // GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
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

            DataManager.GetBatch(_batch, DataManager.Train, _random);

            var handle = NeuralJobs.ZeroGradients(_gradientsAvg);

            for (int i = 0; i < _batch.Length; i++) {
                handle = NeuralJobs.CopyInput(_inputs, DataManager.Train, _batch[i], handle);
                handle = NeuralJobs.ForwardPass(_net, _inputs, handle);

                int lbl = DataManager.Train.Labels[_batch[i]];
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
                int lbl = DataManager.Test.Labels[i];

                var handle = NeuralJobs.CopyInput(_inputs, DataManager.Test, i);
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