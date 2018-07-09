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
using NeuralJobs;

public struct NativeNetwork : System.IDisposable {
    public NativeArray<float> Weights;
    public NativeArray<float> Bias;
    public NativeArray<float> Output;

    public NativeNetwork(int numInputs, int numHidden) {
        Weights = new NativeArray<float>(numHidden * numInputs, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Bias = new NativeArray<float>(numHidden, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        Output = new NativeArray<float>(numHidden, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
    }

    public void Dispose() {
        Weights.Dispose();
        Bias.Dispose();
        Output.Dispose();
    }
}

public class JobTest : MonoBehaviour {
    System.Random _random;

    private void Awake() {
        //Mnist.Load();

        _random = new System.Random(1234);

        const int numInputs = 16;
        const int numHidden = 4;

        var net = new NativeNetwork(numInputs, numHidden);
        Init(_random, net);

        var input = new NativeArray<float>(numInputs, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        RandomGaussian(_random, input, 0f, 1f);

        /* Forward Pass */

        var forwardHandle = ScheduleForwardPass(net);
        forwardHandle.Complete();

        for (int i = 0; i < net.Output.Length; i++) {
            Debug.Log("" + i + ": " + net.Output[i]);
        }

        net.Dispose();
    }

    private static void Init(System.Random random, NativeNetwork net) {
        // Todo: init as jobs too. Needs Burst-compatible RNG.
        
        // var initInputJob = new GaussianJob();
        // initInputJob.Random = new MTRandom(0);
        // initInputJob.T = bias;
        // initInputJob.Mean = 0.5f;
        // initInputJob.Std = 0.5f;

        // var initBiasJob = new GaussianJob();
        // initBiasJob.Random = new MTRandom(1);
        // initBiasJob.T = bias;
        // initBiasJob.Mean = 0f;
        // initBiasJob.Std = 1f;

        // var initWeightsJob = new GaussianJob();
        // initWeightsJob.Random = new MTRandom(2);
        // initWeightsJob.T = weights;
        // initWeightsJob.Mean = 0f;
        // initWeightsJob.Std = 1f;

        // initInputJob.Schedule().Complete();
        // initBiasJob.Schedule().Complete();
        // initWeightsJob.Schedule().Complete();

        RandomGaussian(random, net.Weights, 0f, 1f);
        RandomGaussian(random, net.Bias, 0f, 1f);
    }

    private static JobHandle ScheduleForwardPass(NativeNetwork net, NativeArray<float> input) {
        var addBiasJob = new CopyToJob();
        addBiasJob.A = net.Bias;
        addBiasJob.T = net.Output;
        var addBiasHandle = addBiasJob.Schedule();

        var dotJob = new DotJob();
        dotJob.Input = input;
        dotJob.Weights = net.Weights;
        dotJob.Output = net.Output;

        var dotHandle = dotJob.Schedule(addBiasHandle);

        var sigmoidJob = new SigmoidJob();
        sigmoidJob.A = net.Output;
        var sigmoidHandle = sigmoidJob.Schedule(dotHandle);

        return sigmoidHandle;
    }

    private static void RandomGaussian(System.Random random, NativeArray<float> values, float mean, float std) {
        for (int i = 0; i < values.Length; i++) {
            values[i] = Old.Utils.Gaussian(random, mean, std);
        }
    }
}