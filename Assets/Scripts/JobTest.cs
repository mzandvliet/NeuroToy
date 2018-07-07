using UnityEngine;
using Unity.Jobs;
using Unity.Collections;

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

 */

public class JobTest : MonoBehaviour {
    private void Awake() {
        var a = new NativeArray<float>(16, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var b = new NativeArray<float>(16, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var r = new NativeArray<float>(16, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        for (int i = 0; i < 16; i++) {
            a[i] = i;
            b[i] = 16 - i;
        }

        var j = new AddJob();
        j.A = a;
        j.B = b;
        j.R = r;
        var h = j.Schedule();
        h.Complete();
        for (int i = 0; i < 16; i++) {
            Debug.Log(r[i]);
        }

        a.Dispose();
        b.Dispose();
        r.Dispose();
    }
}