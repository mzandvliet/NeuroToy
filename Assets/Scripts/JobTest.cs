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

public class JobTest : MonoBehaviour {
    private void Awaken() {
        //Mnist.Load();

        /* Allocation */

        const int numInputs = 16;
        const int numHidden = 4;

        var input = new NativeArray<float>(numInputs, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var weights = new NativeArray<float>(numHidden * numInputs, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var bias = new NativeArray<float>(numHidden, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var output = new NativeArray<float>(numHidden, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        /* Init */

        var initInputJob = new GaussianJob();
        initInputJob.Random = new MTRandom(0);
        initInputJob.T = bias;
        initInputJob.Mean = 0.5f;
        initInputJob.Std = 0.5f;
        
        var initBiasJob = new GaussianJob();
        initBiasJob.Random = new MTRandom(1);
        initBiasJob.T = bias;
        initBiasJob.Mean = 0f;
        initBiasJob.Std = 1f;
        
        var initWeightsJob = new GaussianJob();
        initWeightsJob.Random = new MTRandom(2);
        initWeightsJob.T = weights;
        initWeightsJob.Mean = 0f;
        initWeightsJob.Std = 1f;
        
        initInputJob.Schedule().Complete();
        initBiasJob.Schedule().Complete();
        initWeightsJob.Schedule().Complete();

        /* Forward Pass */

        var addBiasJob = new CopyToJob();
        addBiasJob.A = bias;
        addBiasJob.T = output;
        var addBiasHandle = addBiasJob.Schedule();

        var dotJob = new DotJob();
        dotJob.Input = input;
        dotJob.Weights = weights;
        dotJob.Output = output;
        
        var dotHandle = dotJob.Schedule(addBiasHandle);

        var sigmoidJob = new SigmoidJob();
        sigmoidJob.A = output;
        var sigmoidHandle = sigmoidJob.Schedule(dotHandle);

        input.Dispose();
        weights.Dispose();
        bias.Dispose();
        output.Dispose();
    }
}