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
using NeuralJobs;
using Mnist = New.Mnist;

public class JobTest : MonoBehaviour {
    System.Random _random;

    private NativeNetwork _net;
    private NativeOptimizer _optimizer;
    private NativeOptimizer _gradientBucket;

    int _epoch;
    int _batch;
    float _trainingLoss;
    float _rate;

    private void Awake() {
        Application.runInBackground = true;
        Mnist.Load();

        _random = new System.Random(1234);

        var config = new NativeNetworkConfig();
        config.Layers.Add(new NativeLayerConfig { Neurons = Mnist.Train.ImgDims });
        config.Layers.Add(new NativeLayerConfig { Neurons = 30 });
        config.Layers.Add(new NativeLayerConfig { Neurons = 10 });

        _net = new NativeNetwork(config);
        Initialize(_random, _net);
        
        _optimizer = new NativeOptimizer(config);
        _gradientBucket = new NativeOptimizer(config);
    }
    
    private void Update() {
        if (_epoch < 30) {
            if (_batch < 6000) {
                for (int i = 0; i < 10; i++) {
                    TrainMinibatch();
                }
            } else {
                Test();
                _batch = 0;
                _epoch++;
            }
        }
    }

    private void OnGUI() {
        GUILayout.BeginVertical(GUI.skin.box);
        {
            GUILayout.Label("Epoch: " + _epoch);
            GUILayout.Label("Batch: " + _batch);
            GUILayout.Label("Train Loss: " + _trainingLoss);
            GUILayout.Label("Rate: " + _rate);
        }
        GUILayout.EndVertical();

        // GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _label);
        // GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _tex, ScaleMode.ScaleToFit);
    }

    private void OnDestroy() {
        Mnist.Unload();
        _net.Dispose();
        _optimizer.Dispose();
        _gradientBucket.Dispose();
    }

    private static void Initialize(System.Random random, NativeNetwork net) {
        // Todo: init as jobs too. Needs Burst-compatible RNG.

        for (int l = 0; l < net.Layers.Length; l++) {
            RandomGaussian(random, net.Layers[l].Weights, 0f, 1f);
            RandomGaussian(random, net.Layers[l].Biases, 0f, 1f);
        }
    }

    private static void RandomGaussian(System.Random random, NativeArray<float> values, float mean, float std) {
        for (int i = 0; i < values.Length; i++) {
            values[i] = Old.Utils.Gaussian(random, mean, std);
        }
    }

    private void TrainMinibatch() {
        UnityEngine.Profiling.Profiler.BeginSample("TrainMiniBatch");

        const int numClasses = 10;
        const int batchSize = 10;

        var target = new NativeArray<float>(numClasses, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var dCdO = new NativeArray<float>(numClasses, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var input = new NativeArray<float>(Mnist.Test.ImgDims, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        float avgTrainCost = 0f;
        int correctTrainLabels = 0;

        // ResetOptimizer(_gradientBucket); // Todo: need avgGradient thing
        var handle = ScheduleZeroGradients(_gradientBucket, new JobHandle());
        handle.Complete();

        var trainBatch = Mnist.GetBatch(batchSize, Mnist.Train, _random);
        
        for (int i = 0; i < trainBatch.Indices.Length; i++) {
            int lbl = Mnist.Train.Labels[trainBatch.Indices[i]];

            var copyInputJob = new CopySubsetJob();
            copyInputJob.A = Mnist.Test.Images;
            copyInputJob.B = input;
            copyInputJob.ALength = Mnist.Test.ImgDims;
            copyInputJob.AStart = i * Mnist.Test.ImgDims;
            copyInputJob.BStart = 0;
            handle = copyInputJob.Schedule();

            handle = ScheduleForwardPass(_net, input, handle);
            handle.Complete();

            // Todo: better if we don't Complete here, but chain backprop pass

            int predictedLbl = GetMaxOutput(_net.Last.Outputs);
            LabelToOneHot(lbl, target);

            if (predictedLbl == lbl) {
                correctTrainLabels++;
            }
            //Debug.Log(outputClass + ", " + batch.Labels[i]);

            // Calculate error between output layer and target
            Subtract(target, _net.Last.Outputs, dCdO);
            float cost = Cost(dCdO);
            avgTrainCost += cost;

            // Propagate error back
            // Calculate per-parameter gradient, store it

            handle = ScheduleBackwardsPass(_net, _optimizer, input, target, handle);
            handle = ScheduleAddGradients(_optimizer, _gradientBucket, handle);

            handle.Complete();
        }

        avgTrainCost /= (float)batchSize;

        // Update weights and biases according to averaged gradient and learning rate
        _rate = 3.0f / (float)batchSize;
        handle = ScheduleUpdateParameters(_net, _gradientBucket, _rate, new JobHandle());
        handle.Complete();

        _batch++;
        _trainingLoss = (float)System.Math.Round(avgTrainCost, 6);

        target.Dispose();
        dCdO.Dispose();
        input.Dispose();

        // Debug.Log(
        //     "Batch: " + _batchesTrained +
        //     ", TrainLoss: " + Math.Round(avgTrainCost, 6) +
        //     ", Rate: " + Math.Round(rate, 6));
        // Mnist.ToTexture(batch, batch.Labels.Length-1, _tex);
        // _label = batch.Labels[batch.Labels.Length-1];

        UnityEngine.Profiling.Profiler.EndSample();
    }

    private void Test() {
        UnityEngine.Profiling.Profiler.BeginSample("Test");

        NativeArray<float> input = new NativeArray<float>(Mnist.Test.ImgDims, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        int correctTestLabels = 0;
        for (int i = 0; i < Mnist.Test.NumImgs; i++) {
            int lbl = Mnist.Test.Labels[i];

            var copyInputJob = new CopySubsetJob();
            copyInputJob.A = Mnist.Test.Images;
            copyInputJob.B = input;
            copyInputJob.ALength = Mnist.Test.ImgDims;
            copyInputJob.AStart = i * Mnist.Test.ImgDims;
            copyInputJob.BStart = 0;
            var handle = copyInputJob.Schedule();

            handle = ScheduleForwardPass(_net, input, handle);
            handle.Complete();

            int predictedLbl = GetMaxOutput(_net.Last.Outputs);
            if (predictedLbl == lbl) {
                correctTestLabels++;
            }
        }

        float accuracy = correctTestLabels / (float)Mnist.Test.NumImgs;
        Debug.Log("Test Accuracy: " + System.Math.Round(accuracy * 100f, 4) + "%");

        input.Dispose();

        UnityEngine.Profiling.Profiler.EndSample();
    }

    private static void Subtract(NativeArray<float> a, NativeArray<float> b, NativeArray<float> result) {
        if (a.Length != b.Length) {
            throw new System.ArgumentException("Lengths of arrays have to match");
        }

        for (int i = 0; i < a.Length; i++) {
            result[i] = a[i] - b[i];
        }
    }

    private static void LabelToOneHot(int label, NativeArray<float> vector) {
        for (int i = 0; i < vector.Length; i++) {
            vector[i] = i == label ? 1f : 0f;
        }
    }

    private static int GetMaxOutput(NativeArray<float> data) {
        float largestActivation = float.MinValue;
        int idx = 0;
        for (int i = 0; i < data.Length; i++) {
            if (data[i] > largestActivation) {
                largestActivation = data[i];
                idx = i;
            }
        }
        return idx;
    }

    private static float Cost(NativeArray<float> vector) {
        float sum = 0f;
        for (int i = 0; i < vector.Length; i++) {
            sum += vector[i] * vector[i];
        }
        return Unity.Mathematics.math.sqrt(sum);
    }

    private static JobHandle ScheduleForwardPass(NativeNetwork net, NativeArray<float> input, JobHandle handle) {
        NativeArray<float> lastOut = input;

        for (int l = 0; l < net.Layers.Length; l++) {
            var layer = net.Layers[l];

            var addBiasJob = new CopyJob();
            addBiasJob.From = layer.Biases;
            addBiasJob.To = layer.Outputs;
            handle = addBiasJob.Schedule(handle);

            var dotJob = new DotJob();
            dotJob.Input = lastOut;
            dotJob.Weights = layer.Weights;
            dotJob.Output = layer.Outputs;
            handle = dotJob.Schedule(handle);

            var sigmoidJob = new SigmoidEqualsJob();
            sigmoidJob.A = layer.Outputs;
            handle = sigmoidJob.Schedule(handle);

            lastOut = layer.Outputs;
        }

        return handle;
    }

    private static JobHandle ScheduleBackwardsPass(NativeNetwork net, NativeOptimizer optimizer, NativeArray<float> input, NativeArray<float> target, JobHandle handle) {
        JobHandle h = handle;

        var subtractJob = new SubtractJob();
        subtractJob.A = net.Last.Outputs;
        subtractJob.B = target;
        subtractJob.T = optimizer.DCDO;
        h = subtractJob.Schedule(h);

        var backwardsFinalJob = new BackwardsFinalJob();
        backwardsFinalJob.DCDO = optimizer.DCDO;
        backwardsFinalJob.DCDZ = optimizer.Last.DCDZ;
        backwardsFinalJob.DCDW = optimizer.Last.DCDW;
        backwardsFinalJob.Output = net.Last.Outputs;
        backwardsFinalJob.OutputsPrev = net.Layers[net.Layers.Length-2].Outputs;
        h = backwardsFinalJob.Schedule(h);

        for (int l = net.Layers.Length - 2; l > 0; l--) {
            var backwardsJob = new BackwardsJob();
            backwardsJob.DCDZNext = optimizer.Layers[l+1].DCDZ;
            backwardsJob.WeightsNext = net.Layers[l+1].Weights;
            backwardsJob.DCDZ = optimizer.Layers[l].DCDZ;
            backwardsJob.DCDW = optimizer.Layers[l].DCDW;
            backwardsJob.Output = net.Layers[l].Outputs;
            backwardsJob.OutputsPrev = l == 1 ? input : net.Layers[l - 1].Outputs;
            h = backwardsJob.Schedule(h);
        }

        return h;
    }

    // Todo: jobify
    // private static void ResetOptimizer(NativeOptimizer optimizer) {
    //     UnityEngine.Profiling.Profiler.BeginSample("ResetOptimizer");

    //     for (int l = 0; l < optimizer.Layers.Length; l++) {
    //         var dcdz = optimizer.Layers[l].DCDZ;

    //         for (int n = 0; n < dcdz.Length; n++) {
    //             dcdz[n] = 0f;
                
    //             var dcdw = optimizer.Layers[l].DCDW;
    //             for (int w = 0; w < dcdw.Length; w++) {
    //                 optimizer.Layers[l].DCDW[w] = 0f;
    //             }
    //         }
    //     }

    //     UnityEngine.Profiling.Profiler.EndSample();
    // }

    private static JobHandle ScheduleZeroGradients(NativeOptimizer o, JobHandle handle) {
        // Todo: parallelize over layers and/or biases/weights
        for (int l = 0; l < o.Layers.Length; l++) {
            var setBiasJob = new SetValueJob();
            setBiasJob.A = o.Layers[l].DCDZ;
            setBiasJob.Scalar = 0f;
            handle = setBiasJob.Schedule(handle);

            var setWeightsJob = new SetValueJob();
            setWeightsJob.A = o.Layers[l].DCDW;
            setWeightsJob.Scalar = 0f;
            handle = setWeightsJob.Schedule(handle);
        }

        return handle;
    }

    private static JobHandle ScheduleAddGradients(NativeOptimizer a, NativeOptimizer b, JobHandle handle) {
        // Todo: parallelize over layers and/or biases/weights
        for (int l = 0; l < a.Layers.Length; l++) {
            var addBiasJob = new AddEqualsJob();
            addBiasJob.A = a.Layers[l].DCDZ;
            addBiasJob.B = b.Layers[l].DCDZ;
            handle = addBiasJob.Schedule(handle);

            var addWeightsJob = new AddEqualsJob();
            addWeightsJob.A = a.Layers[l].DCDW;
            addWeightsJob.B = b.Layers[l].DCDW;
            handle = addWeightsJob.Schedule(handle);
        }

        return handle;
    }

    private static JobHandle ScheduleUpdateParameters(NativeNetwork net, NativeOptimizer gradients, float rate, JobHandle handle) {
        // Todo: Find a nice way to fold the mult by learning rate and addition together in one job

        for (int l = 0; l < net.Layers.Length; l++) {
            var m = new MultiplyEqualsJob();
            m.A = gradients.Layers[l].DCDZ;
            m.Scalar = rate;
            handle = m.Schedule(handle);

            var s = new SubtractEqualsJob();
            s.A = gradients.Layers[l].DCDZ;
            s.B = net.Layers[l].Biases;
            handle = s.Schedule(handle);

            m = new MultiplyEqualsJob();
            m.A = gradients.Layers[l].DCDW;
            m.Scalar = rate;
            handle = m.Schedule(handle);

            s = new SubtractEqualsJob();
            s.A = gradients.Layers[l].DCDW;
            s.B = net.Layers[l].Weights;
            handle = s.Schedule(handle);
        }

        return handle;
    }
}

public struct NativeLayerConfig {
    public int Neurons;
}

public class NativeNetworkConfig {
    public List<NativeLayerConfig> Layers;

    public NativeNetworkConfig() {
        Layers = new List<NativeLayerConfig>();
    }
}

// Todo: could test layers just existing of slices of single giant arrays
public class NativeNetworkLayer : System.IDisposable {
    public NativeArray<float> Biases;
    public NativeArray<float> Weights;
    public NativeArray<float> Outputs;
    public readonly int NumNeurons;
    public readonly int NumInputs;

    public NativeNetworkLayer(int numNeurons, int numInputs) {
        Biases = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Weights = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Outputs = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        NumNeurons = numNeurons;
        NumInputs = numInputs;
    }

    public void Dispose() {
        Biases.Dispose();
        Weights.Dispose();
        Outputs.Dispose();
    }
}

public class NativeNetwork : System.IDisposable {
    public NativeNetworkLayer[] Layers;

    public NativeNetworkLayer Last {
        get { return Layers[Layers.Length - 1]; }
    }

    public NativeNetwork(NativeNetworkConfig config) {
        Layers = new NativeNetworkLayer[config.Layers.Count - 1];
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l] = new NativeNetworkLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
        }
    }

    public void Dispose() {
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l].Dispose();
        }
    }
}

public class NativeOptimizerLayer : System.IDisposable {
    public NativeArray<float> DCDZ;
    public NativeArray<float> DCDW;
    public readonly int NumNeurons;
    public readonly int NumInputs;

    public NativeOptimizerLayer(int numNeurons, int numInputs) {
        DCDZ = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        DCDW = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        NumNeurons = numNeurons;
        NumInputs = numInputs;
    }

    public void Dispose() {
        DCDZ.Dispose();
        DCDW.Dispose();
    }
}

public class NativeOptimizer : System.IDisposable {
    public NativeOptimizerLayer[] Layers;
    public NativeArray<float> DCDO;
    public NativeNetworkConfig Config;

    public NativeOptimizerLayer Last {
        get { return Layers[Layers.Length - 1]; }
    }

    public NativeOptimizer(NativeNetworkConfig config) {
        Layers = new NativeOptimizerLayer[config.Layers.Count - 1];
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l] = new NativeOptimizerLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
        }

        DCDO = new NativeArray<float>(config.Layers[config.Layers.Count - 1].Neurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Config = config;
    }

    public void Dispose() {
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l].Dispose();
        }
        DCDO.Dispose();
    }
}