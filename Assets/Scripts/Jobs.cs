using Unity.Jobs;
using Unity.Collections;
using UnityEngine;

public struct AddJob : IJob {
    public NativeArray<float> A;
    public NativeArray<float> B;
    public NativeArray<float> Result;

    public void Execute() {
        for (int i = 0; i < A.Length; i++) {
            Result[i] = A[i] + B[i];
        }
    }
}