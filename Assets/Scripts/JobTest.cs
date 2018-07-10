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

    private void Awake() {
        Mnist.Load();

        _random = new System.Random(1234);

        var config = new NativeNetworkConfig();
        config.Layers.Add(new NativeLayerConfig { Neurons = Mnist.Train.ImgDims });
        config.Layers.Add(new NativeLayerConfig { Neurons = 30 });
        config.Layers.Add(new NativeLayerConfig { Neurons = 10 });

        _net = new NativeNetwork(config);
        Initialize(_random, _net);
        for (int i = 0; i < _net.Layers.Length; i++) {
            Debug.Log(_net.Layers[i].Outputs.Length + ", " + _net.Layers[i].Weights.Length);
        }
        
        _optimizer = new NativeOptimizer(config);
    }
    
    private void Update() {
        if (Input.GetKeyDown(KeyCode.T)) {
            Test();
        }
    }

    private void OnDestroy() {
        Mnist.Unload();
        _net.Dispose();
        _optimizer.Dispose();
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

    private void Test() {
        UnityEngine.Profiling.Profiler.BeginSample("Test");

        NativeArray<float> input = new NativeArray<float>(Mnist.Test.ImgDims, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        int correctTestLabels = 0;
        for (int i = 0; i < Mnist.Test.NumImgs; i++) {
            int lbl = Mnist.Test.Labels[i];

            var copyInputJob = new CopyInputJob();
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

    private static JobHandle ScheduleForwardPass(NativeNetwork net, NativeArray<float> input, JobHandle handle) {
        NativeArray<float> lastOut = input;

        for (int l = 0; l < net.Layers.Length; l++) {
            var layer = net.Layers[l];

            var addBiasJob = new CopyToJob();
            addBiasJob.A = layer.Biases;
            addBiasJob.T = layer.Outputs;
            handle = addBiasJob.Schedule(handle);

            var dotJob = new DotJob();
            dotJob.Input = lastOut;
            dotJob.Weights = layer.Weights;
            dotJob.Output = layer.Outputs;
            handle = dotJob.Schedule(handle);

            var sigmoidJob = new SigmoidJob();
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

    public NativeNetworkLayer(int numNeurons, int numInputs) {
        Biases = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Weights = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Outputs = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
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

    public NativeOptimizerLayer(int numNeurons, int numInputs) {
        DCDZ = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        DCDW = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.ClearMemory);
    }

    public void Dispose() {
        DCDZ.Dispose();
        DCDW.Dispose();
    }
}

public class NativeOptimizer : System.IDisposable {
    public NativeOptimizerLayer[] Layers;
    public NativeArray<float> DCDO;

    public NativeOptimizerLayer Last {
        get { return Layers[Layers.Length - 1]; }
    }

    public NativeOptimizer(NativeNetworkConfig config) {
        Layers = new NativeOptimizerLayer[config.Layers.Count - 1];
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l] = new NativeOptimizerLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
        }

        DCDO = new NativeArray<float>(config.Layers[config.Layers.Count - 1].Neurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
    }

    public void Dispose() {
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l].Dispose();
        }
        DCDO.Dispose();
    }
}