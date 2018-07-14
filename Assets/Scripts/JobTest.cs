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

        Mnist.Load();

        _random = new System.Random();

        var config = new NativeNetworkConfig();
        config.Layers.Add(new NativeLayerConfig { Neurons = Mnist.Train.ImgDims });
        config.Layers.Add(new NativeLayerConfig { Neurons = 30 });
        config.Layers.Add(new NativeLayerConfig { Neurons = 10 });

        _net = new NativeNetwork(config);
        Initialize(_random, _net);

        _gradients = new NativeGradients(config);
        _gradientsAvg = new NativeGradients(config);
        _batch = new NativeArray<int>(BatchSize, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        _targetOutputs = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _dCdO = new NativeArray<float>(OutputClassCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _inputs = new NativeArray<float>(Mnist.Test.ImgDims, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        _watch = System.Diagnostics.Stopwatch.StartNew();
    }
    
    private void Update() {
        if (_epochCount < 30) {
            if (_batchCount < 6000) {
                for (int i = 0; i < 100; i++) {
                    TrainMinibatch();
                }
            } else {
                Test();
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
            GUILayout.Label("Batch: " + _batchCount);
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
        _gradients.Dispose();
        _gradientsAvg.Dispose();

        _batch.Dispose();
        _targetOutputs.Dispose();
        _dCdO.Dispose();
        _inputs.Dispose();
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

        float avgTrainCost = 0f;

        var handle = ScheduleZeroGradients(_gradientsAvg, new JobHandle());
        handle.Complete(); // Todo: is this needed?

        Mnist.GetBatch(_batch, Mnist.Train, _random);
        
        for (int i = 0; i < _batch.Length; i++) {
            int lbl = Mnist.Train.Labels[_batch[i]];

            handle = ScheduleCopyInput(_inputs, Mnist.Train, _batch[i], handle);
            handle = ScheduleForwardPass(_net, _inputs, handle);

            ClassToOneHot(lbl, _targetOutputs); // Todo: job
            handle = ScheduleBackwardsPass(_net, _gradients, _inputs, _targetOutputs, handle);
            handle = ScheduleAddGradients(_gradients, _gradientsAvg, handle);
            handle.Complete();

            // Todo: backwards pass logic now does this, don't redo, just check
            Subtract(_targetOutputs, _net.Last.Outputs, _dCdO);
            float cost = Cost(_dCdO);
            avgTrainCost += cost;
        }

        avgTrainCost /= (float)BatchSize;

        // Update weights and biases according to averaged gradient and learning rate
        _rate = 3.0f / (float)BatchSize;
        handle = ScheduleUpdateParameters(_net, _gradientsAvg, _rate, new JobHandle());
        handle.Complete(); // Todo: Is this one needed?

        _batchCount++;
        _trainingLoss = (float)System.Math.Round(avgTrainCost, 6);

        UnityEngine.Profiling.Profiler.EndSample();
    }

    private void Test() {
        UnityEngine.Profiling.Profiler.BeginSample("Test");

        int correctTestLabels = 0;
        for (int i = 0; i < Mnist.Test.NumImgs; i++) { 
            int lbl = Mnist.Test.Labels[i];

            var handle = ScheduleCopyInput(_inputs, Mnist.Test, i);
            handle = ScheduleForwardPass(_net, _inputs, handle);
            handle.Complete();

            int predictedLbl = ArgMax(_net.Last.Outputs);
            if (predictedLbl == lbl) {
                correctTestLabels++;
            }
        }

        float accuracy = correctTestLabels / (float)Mnist.Test.NumImgs;
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

    private static void Subtract(NativeArray<float> a, NativeArray<float> b, NativeArray<float> result) {
        if (a.Length != b.Length) {
            throw new System.ArgumentException("Lengths of arrays have to match");
        }

        for (int i = 0; i < a.Length; i++) {
            result[i] = a[i] - b[i];
        }
    }

    private static void ClassToOneHot(int c, NativeArray<float> vector) {
        for (int i = 0; i < vector.Length; i++) {
            vector[i] = i == c ? 1f : 0f;
        }
    }

    private static int ArgMax(NativeArray<float> data) {
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

    private static JobHandle ScheduleCopyInput(NativeArray<float> inputs, New.Dataset set, int imgIdx, JobHandle handle = new JobHandle()) {
        var copyInputJob = new CopySubsetJob();
        copyInputJob.From = set.Images;
        copyInputJob.To = inputs;
        copyInputJob.Length = set.ImgDims;
        copyInputJob.FromStart = imgIdx * set.ImgDims;
        copyInputJob.ToStart = 0;
        return copyInputJob.Schedule(handle);
    }

    private static JobHandle ScheduleForwardPass(NativeNetwork net, NativeArray<float> input, JobHandle handle = new JobHandle()) {
        NativeArray<float> last = input;

        for (int l = 0; l < net.Layers.Length; l++) {
            var layer = net.Layers[l];

            var b = new CopyParallelJob();
            b.From = layer.Biases;
            b.To = layer.Outputs;
            handle = b.Schedule(layer.Outputs.Length, layer.Outputs.Length / 8, handle);

            var d = new DotParallelJob();
            d.Input = last;
            d.Weights = layer.Weights;
            d.Output = layer.Outputs;
            handle = d.Schedule(layer.Outputs.Length, layer.Outputs.Length / 8, handle);

            var s = new SigmoidEqualsParallelJob();
            s.Data = layer.Outputs;
            handle = s.Schedule(layer.Outputs.Length, layer.Outputs.Length / 8, handle);

            last = layer.Outputs;
        }

        return handle;
    }

    private static JobHandle ScheduleBackwardsPass(NativeNetwork net, NativeGradients gradients, NativeArray<float> input, NativeArray<float> target, JobHandle handle = new JobHandle()) {
        JobHandle h = handle;

        var subtractJob = new SubtractJob();
        subtractJob.A = net.Last.Outputs;
        subtractJob.B = target;
        subtractJob.Output = gradients.DCDO;
        h = subtractJob.Schedule(h);

        var backwardsFinalJob = new BackPropFinalJob();
        backwardsFinalJob.DCDO = gradients.DCDO;
        backwardsFinalJob.DCDZ = gradients.Last.DCDZ;
        backwardsFinalJob.DCDW = gradients.Last.DCDW;
        backwardsFinalJob.Outputs = net.Last.Outputs;
        backwardsFinalJob.OutputsPrev = net.Layers[net.Layers.Length-2].Outputs;
        h = backwardsFinalJob.Schedule(h);

        // Note, indexing using net.layers.length here is misleading, since that count is one less than if you include input layer
        for (int l = net.Layers.Length - 2; l >= 0; l--) {
            var backwardsJob = new BackPropJob();
            backwardsJob.DCDZNext = gradients.Layers[l+1].DCDZ;
            backwardsJob.WeightsNext = net.Layers[l+1].Weights;
            backwardsJob.DCDZ = gradients.Layers[l].DCDZ;
            backwardsJob.DCDW = gradients.Layers[l].DCDW;
            backwardsJob.LOutputs = net.Layers[l].Outputs;
            backwardsJob.OutputsPrev = l == 0 ? input : net.Layers[l - 1].Outputs;
            h = backwardsJob.Schedule(h);
            // h = backwardsJob.Schedule(gradients.Layers[l].NumNeurons, gradients.Layers[l].NumNeurons/8, h);
        }

        return h;
    }

    private static JobHandle ScheduleZeroGradients(NativeGradients gradients, JobHandle handle = new JobHandle()) {
        // Todo: parallelize over layers and/or biases/weights
        for (int l = 0; l < gradients.Layers.Length; l++) {
            var setBiasJob = new SetValueJob();
            setBiasJob.Data = gradients.Layers[l].DCDZ;
            setBiasJob.Value = 0f;
            handle = setBiasJob.Schedule(handle);

            var setWeightsJob = new SetValueJob();
            setWeightsJob.Data = gradients.Layers[l].DCDW;
            setWeightsJob.Value = 0f;
            handle = setWeightsJob.Schedule(handle);
        }

        return handle;
    }

    private static JobHandle ScheduleAddGradients(NativeGradients from, NativeGradients to, JobHandle handle = new JobHandle()) {
        // Todo: parallelize over layers and/or biases/weights
        for (int l = 0; l < from.Layers.Length; l++) {
            var addBiasJob = new AddEqualsJob();
            addBiasJob.Data = from.Layers[l].DCDZ;
            addBiasJob.To = to.Layers[l].DCDZ;
            handle = addBiasJob.Schedule(handle);

            var addWeightsJob = new AddEqualsJob();
            addWeightsJob.Data = from.Layers[l].DCDW;
            addWeightsJob.To = to.Layers[l].DCDW;
            handle = addWeightsJob.Schedule(handle);
        }

        return handle;
    }

    private static JobHandle ScheduleUpdateParameters(NativeNetwork net, NativeGradients gradients, float rate, JobHandle handle = new JobHandle()) {
        // Todo: Find a nice way to fold the multiply by learning rate and addition together in one pass over the data
        // Also, parallelize over all the arrays

        for (int l = 0; l < net.Layers.Length; l++) {
            var m = new MultiplyEqualsJob();
            m.Data = gradients.Layers[l].DCDZ;
            m.Value = rate;
            handle = m.Schedule(handle);

            var s = new SubtractEqualsJob();
            s.Data = gradients.Layers[l].DCDZ;
            s.From = net.Layers[l].Biases;
            handle = s.Schedule(handle);

            m = new MultiplyEqualsJob();
            m.Data = gradients.Layers[l].DCDW;
            m.Value = rate;
            handle = m.Schedule(handle);

            s = new SubtractEqualsJob();
            s.Data = gradients.Layers[l].DCDW;
            s.From = net.Layers[l].Weights;
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
        Biases = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        Weights = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        Outputs = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
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

    public NativeNetworkConfig Config {
        get;
        private set;
    }

    public NativeNetwork(NativeNetworkConfig config) {
        Layers = new NativeNetworkLayer[config.Layers.Count - 1];
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l] = new NativeNetworkLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
        }
        Config = config;
    }

    public void Dispose() {
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l].Dispose();
        }
    }
}

public class NativeGradientsLayer : System.IDisposable {
    public NativeArray<float> DCDZ;
    public NativeArray<float> DCDW;
    public readonly int NumNeurons;
    public readonly int NumInputs;

    public NativeGradientsLayer(int numNeurons, int numInputs) {
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

public class NativeGradients : System.IDisposable {
    public NativeGradientsLayer[] Layers;
    public NativeArray<float> DCDO;
    public NativeNetworkConfig Config;

    public NativeGradientsLayer Last {
        get { return Layers[Layers.Length - 1]; }
    }

    public NativeGradients(NativeNetworkConfig config) {
        Layers = new NativeGradientsLayer[config.Layers.Count - 1];
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l] = new NativeGradientsLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
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