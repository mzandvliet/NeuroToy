using UnityEngine;
using Unity.Jobs;
using Unity.Collections;

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
        j.Result = r;
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