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
public class NativeLayer : System.IDisposable {
    public NativeArray<float> Biases;
    public NativeArray<float> Weights;
    public NativeArray<float> Outputs;

    public NativeLayer(int numNeurons, int numInputs) {
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
    public NativeLayer[] Layers;

    public NativeLayer Last {
        get { return Layers[Layers.Length-1]; }
    }

    public NativeNetwork(NativeNetworkConfig config) {
        Layers = new NativeLayer[config.Layers.Count-1];
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l] = new NativeLayer(config.Layers[l+1].Neurons, config.Layers[l].Neurons);
        }
    }

    public void Dispose() {
        for (int l = 0; l < Layers.Length; l++) {
            Layers[l].Dispose();
        }
    }
}

public class JobTest : MonoBehaviour {
    System.Random _random;

    private void Awake() {
        //Mnist.Load();

        _random = new System.Random(1234);

        var config = new NativeNetworkConfig();
        config.Layers.Add(new NativeLayerConfig { Neurons = 786 });
        config.Layers.Add(new NativeLayerConfig { Neurons = 30 });
        config.Layers.Add(new NativeLayerConfig { Neurons = 10 });

        var net = new NativeNetwork(config);
        Init(_random, net);

        var input = new NativeArray<float>(config.Layers[0].Neurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        RandomGaussian(_random, input, 0f, 1f);

        /* Forward Pass */

        var forwardHandle = ScheduleForwardPass(net, input);
        forwardHandle.Complete();

        for (int i = 0; i < net.Last.Outputs.Length; i++) {
            Debug.Log("" + i + ": " + net.Last.Outputs[i]);
        }

        net.Dispose();
        input.Dispose();
    }

    private static void Init(System.Random random, NativeNetwork net) {
        // Todo: init as jobs too. Needs Burst-compatible RNG.

        for (int l = 0; l < net.Layers.Length; l++) {
            RandomGaussian(random, net.Layers[l].Weights, 0f, 1f);
            RandomGaussian(random, net.Layers[l].Biases, 0f, 1f);
        }
    }

    private static JobHandle ScheduleForwardPass(NativeNetwork net, NativeArray<float> input) {
        JobHandle h = new JobHandle();
        NativeArray<float> lastOut = input;

        for (int l = 0; l < net.Layers.Length; l++) {
            var layer = net.Layers[l];

            var addBiasJob = new CopyToJob();
            addBiasJob.A = layer.Biases;
            addBiasJob.T = layer.Outputs;
            h = addBiasJob.Schedule(h);

            var dotJob = new DotJob();
            dotJob.Input = lastOut;
            dotJob.Weights = layer.Weights;
            dotJob.Output = layer.Outputs;

            h = dotJob.Schedule(h);

            var sigmoidJob = new SigmoidJob();
            sigmoidJob.A = layer.Outputs;
            h = sigmoidJob.Schedule(h);

            lastOut = layer.Outputs;
        }

        return h;
    }

    private static void RandomGaussian(System.Random random, NativeArray<float> values, float mean, float std) {
        for (int i = 0; i < values.Length; i++) {
            values[i] = Old.Utils.Gaussian(random, mean, std);
        }
    }
}